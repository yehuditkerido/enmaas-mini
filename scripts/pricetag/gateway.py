#!/usr/bin/env python3
"""Render the HTTPS spending gateway from an existing native provider config."""
import argparse
import copy
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'common'))
from unified_config import active, catalog, cluster_name, render as unified, validate
from switchyard_config import normalize as normalize_switchyard

INFERENCE = ['/v1/chat/completions', '/v1/responses', '/v1/messages']
DASHBOARD = ['/', '/login', '/logout', '/dashboard', '/admin', '/manager', '/me', '/welcome', '/account',
             '/management', '/admin/impersonate', '/me/keys', '/whoami', '/api/v1/whoami', '/api/v1/pricing',
             '/api/v1/dashboard/overview', '/api/v1/dashboard/groups',
             '/api/v1/dashboard/users', '/api/v1/dashboard/directory-users',
             '/api/v1/dashboard/models', '/api/v1/dashboard/tools',
             '/api/v1/dashboard/geo', '/api/v1/dashboard/cost-centers',
             '/api/v1/dashboard/timeline', '/api/v1/dashboard/recent',
             '/api/v1/me/usage/summary', '/api/v1/me/usage/models', '/api/v1/me/usage/recent',
             '/api/v1/me/keys',
             '/api/v1/admin/safety-net', '/api/v1/admin/models', '/api/v1/admin/partner-users',
             '/api/v1/admin/model-policies', '/api/v1/admin/pricing-discounts',
             '/api/v1/admin/pricing/refresh', '/api/v1/admin/pricing/reprice', '/api/v1/org/scope',
             '/api/v1/org/usage', '/api/v1/org/person', '/api/v1/org/charts']
DASHBOARD_PREFIXES = ['/api/v1/management/', '/me/keys/', '/api/v1/me/keys/', '/api/v1/admin/partner-users/', '/api/v1/admin/pricing/reprice/']


def _public_rate_limit():
    """Return the public listener's request limiter for every public entrypoint."""
    return {'filter': 'rate_limit', 'mode': 'global', 'rate': 30, 'burst': 120}


def _condition_not_alias(alias):
    return {'unless': {'headers': {'X-Gateway-Model': alias}}}


def _filter(chain, name):
    return next((item for item in chain['filters'] if item['filter'] == name), None)


def _switchyard_route(original, settings):
    """Build a Switchyard route using only an existing OpenAI provider route.

    The POC intentionally uses the installed provider's existing clusters and
    credentials.  It neither accepts a new endpoint nor creates a new secret.
    """
    source = next((chain for chain in original['filter_chains'] if chain['name'] == 'openai'), None)
    if source is None:
        raise ValueError('Switchyard requires an OpenAI provider chain')
    load_balancer = _filter(source, 'load_balancer')
    credentials = _filter(source, 'credential_injection')
    if load_balancer is None or credentials is None:
        raise ValueError('Switchyard requires existing OpenAI clusters and credentials')
    clusters = {cluster['name']: cluster for cluster in load_balancer['clusters']}
    injected = {cluster['name']: cluster for cluster in credentials['clusters']}

    def target(name):
        model = settings[name]
        cluster = cluster_name(model, 'openai')
        if cluster not in clusters or cluster not in injected:
            raise ValueError('Switchyard ' + name + ' is not an active OpenAI provider model')
        return cluster, model

    judge_cluster, judge = target('judge')
    backend = clusters[judge_cluster]
    credential = injected[judge_cluster]
    if len(backend.get('endpoints', [])) != 1:
        raise ValueError('POC Switchyard judge requires one configured upstream endpoint')
    if credential.get('header') != 'Authorization' or credential.get('header_prefix') != 'Bearer ':
        raise ValueError('POC Switchyard judge requires an OpenAI Bearer credential')
    authority = backend.get('http', {}).get('authority', backend['endpoints'][0])
    scheme = 'https' if backend.get('tls') else 'http'
    judge_config = {
        'endpoint': f'{scheme}://{authority}/v1/chat/completions',
        'model': judge['model'],
        'auth': {'value_env': credential['env_var']},
        'timeout_ms': 10000,
        'verify_tls': bool(backend.get('tls')),
    }
    weak_cluster, weak = target('weak')
    strong_cluster, strong = target('strong')
    return {
        'filter': 'switchyard_route',
        'judge': judge_config,
        'threshold': 0.8,
        'targets': {
            'weak': {'cluster': weak_cluster, 'model': weak['model']},
            'strong': {'cluster': strong_cluster, 'model': strong['model']},
        },
        'on_failure': 'closed',
        'session_floor': 'disabled',
    }


def _metering_filter():
    return {'filter': 'external_metering',
            'metering_url': 'http://pricetag-metering:8080', 'allow_private_endpoint': True,
            'internal_auth_file': '/run/secrets/pricetag-m2m',
            'fail_open': False, 'timeout_seconds': 5, 'source': 'secure-single-server',
            'default_username': '', 'default_model': 'unknown',
            'model_policy_check': True,
            'identity_header_prefix': 'x-tenant-', 'feature_key': 'inference-tokens'}


def _switchyard_fallback(settings, route, api_key_auth, identity_guard, meter, headers, credentials, load_balancer):
    """Render the bounded, unmetered free fallback proven by the demo.

    Iterative-router steps get fresh request contexts.  Repeat authentication
    and identity propagation in the metered primary step so external metering
    sees the ordinary caller rather than an anonymous sub-request.
    """
    fallback_cluster = cluster_name(settings['fallback_model'], 'openai')
    credential_names = {item['name'] for item in credentials['clusters']}
    backend_names = {item['name'] for item in load_balancer['clusters']}
    if fallback_cluster not in credential_names or fallback_cluster not in backend_names:
        raise ValueError('Switchyard fallback_model is not an active OpenAI provider model')
    primary = [copy.deepcopy(api_key_auth), copy.deepcopy(identity_guard), route, meter,
               {'filter': 'token_count', 'provider': 'openai'}]
    free = [
        {'filter': 'switchyard_failure_target', 'cluster': fallback_cluster,
         'model': settings['fallback_model']['model']},
        {'filter': 'token_count', 'provider': 'openai'},
    ]
    # Nested iterative-router steps do not inherit request credentials or
    # upstream clusters.  Copy the deployed provider settings into both.
    for step in (primary, free):
        step += [copy.deepcopy(headers), copy.deepcopy(credentials), copy.deepcopy(load_balancer)]
    return {
        'filter': 'iterative_request_router',
        'initial_step': 'primary',
        'max_iterations': 2,
        'steps': [
            {'name': 'primary', 'filters': primary,
             'on_result': [
                 {'origin': 'local', 'status': [429], 'next': 'free'},
                 {'default': True, 'done': True},
             ]},
            {'name': 'free', 'filters': free,
             'on_result': [{'default': True, 'done': True}]},
        ],
    }


def _switchyard_openai_chain(openai, original, settings, public_catalog):
    """Create the isolated Chat-Completions listener for ``autorouter``.

    Praxis permits only one cluster-selecting filter in a pipeline.  Conditions
    do not make a conditional ``switchyard_route`` compatible with the ordinary
    ``intelligent_route``.  Preserve that direct-model chain and derive the
    isolated listener used by the existing all-in-one Switchyard profile.
    """
    alias = settings['alias']
    filters = copy.deepcopy(openai['filters'])
    access = next((item for item in filters if item['filter'] == 'model_access'), None)
    unknown = next((item for item in filters
                    if item['filter'] == 'static_response' and item.get('body') == 'Unknown model'), None)
    unsupported = next((item for item in filters
                        if item['filter'] == 'static_response' and item.get('body') == 'Unsupported API path'), None)
    catalog_response = next((item for item in filters
                             if item['filter'] == 'static_response' and item.get('status') == 200), None)
    route_index = next((i for i, item in enumerate(filters)
                        if item['filter'] == 'intelligent_route'), None)
    if None in (access, unknown, unsupported, catalog_response, route_index):
        raise ValueError('Switchyard requires the standard OpenAI gateway pipeline')
    # 8444 is a public HTTPS listener too.  Retain the same request-rate
    # protection as the direct-model entrypoint before it can route upstream.
    identity_index = next((i for i, item in enumerate(filters)
                           if item['filter'] == 'identity_header_guard'), None)
    if identity_index is None:
        raise ValueError('Switchyard requires the standard OpenAI identity boundary')
    filters.insert(identity_index + 1, _public_rate_limit())
    route_index += 1
    # model_access observes the rewritten body after Switchyard has chosen
    # Weak/Strong.  Permit that small internal target set as well as the
    # caller-visible alias; the following Unknown-model filter still rejects
    # a client that tries to submit one of those target names to port 8444.
    access['models'] = sorted({alias, settings['weak']['model'], settings['strong']['model']} |
                              ({settings['fallback_model']['model']} if settings['fallback'] else set()))
    # Keep the original public-alias boundary.  This guard runs before the
    # Switchyard route rewrites the JSON body, so it must *not* answer for
    # autorouter itself; it rejects any other caller-supplied model name.
    unknown['conditions'] = [_condition_not_alias(alias)]
    unsupported['conditions'] = [{'unless': {'path': '/v1/chat/completions'}}]
    route = _switchyard_route(original, settings)
    chat_only = [{'when': {'path': '/v1/chat/completions'}}]
    if settings['fallback']:
        api_key_auth = next((item for item in filters if item['filter'] == 'api_key_auth'), None)
        identity_guard = next((item for item in filters if item['filter'] == 'identity_header_guard'), None)
        token_count = next((item for item in filters[route_index:]
                            if item['filter'] == 'token_count'), None)
        credentials = next((item for item in filters[route_index:]
                            if item['filter'] == 'credential_injection'), None)
        load_balancer = next((item for item in filters[route_index:]
                              if item['filter'] == 'load_balancer'), None)
        meter = next((item for item in filters[route_index:]
                      if item['filter'] == 'external_metering'), None)
        headers = next((item for item in filters if item['filter'] == 'headers'), None)
        if None in (api_key_auth, identity_guard, token_count, credentials, load_balancer, meter, headers):
            raise ValueError('Switchyard fallback requires the OpenAI metering pipeline')
        fallback = _switchyard_fallback(settings, route, api_key_auth, identity_guard,
                                        copy.deepcopy(meter), headers, credentials, load_balancer)
        fallback['conditions'] = chat_only
        filters[route_index:] = [fallback]
    else:
        route['conditions'] = chat_only
        filters[route_index] = route
    # The standard 8443 catalog intentionally advertises direct routes only.
    # This authenticated 8444 catalog advertises the isolated alias.
    catalog_response['body'] = json.dumps(public_catalog)
    return {'name': 'switchyard-openai', 'filters': filters}


def render(original, models, origins=None, switchyard=None):
    models = validate(models)
    public_catalog = catalog(models, 'openai')
    settings = normalize_switchyard(switchyard, models) if switchyard is not None else None
    switchyard_catalog = copy.deepcopy(public_catalog)
    if settings is not None:
        switchyard_catalog['data'].append({
            'id': settings['alias'], 'object': 'model', 'created': 0, 'owned_by': 'switchyard',
            'praxis': {'routing': 'switchyard', 'api': 'openai-chat-completions'},
        })
    for entry, model in zip(public_catalog['data'], models):
        entry['praxis']['apis'] = model['apis']
    result = unified(original, models, intelligent_routing=True)
    result['admin'] = {'address': '127.0.0.1:9901'}
    result['insecure_options'] = {'allow_private_endpoints': True, 'allow_private_upstreams': True}
    result['listeners'] = [
        {'name': 'https', 'address': '0.0.0.0:8443', 'filter_chains': ['edge'],
         'tls': {'certificates': [{'cert_path': '/etc/praxis/tls.pem',
                                  'key_path': '/etc/praxis/tls-key.pem'}]}},
        {'name': 'authenticated-openai', 'address': '127.0.0.1:18082', 'filter_chains': ['openai']},
        {'name': 'authenticated-anthropic', 'address': '127.0.0.1:18083', 'filter_chains': ['anthropic']},
        {'name': 'dashboard-boundary', 'address': '127.0.0.1:18085', 'filter_chains': ['dashboard']}]
    if settings is not None:
        result['listeners'].append({
            'name': 'switchyard-openai', 'address': '0.0.0.0:8444', 'filter_chains': ['switchyard-openai'],
            'tls': {'certificates': [{'cert_path': '/etc/praxis/tls.pem',
                                      'key_path': '/etc/praxis/tls-key.pem'}]}})
    for chain in result['filter_chains']:
        api = chain['name']
        filters = [f for f in chain['filters'] if f['filter'] not in ('token_rate_limit', 'policy', 'rate_limit')]
        request_headers = next((f for f in filters if f['filter'] == 'headers'), None)
        if request_headers is not None:
            filters.remove(request_headers)
        key_filter = {'filter': 'api_key_auth',
            'validate_url': 'http://pricetag-maas:8080/internal/v1/api-keys/validate',
            'token_header': 'authorization' if api == 'openai' else 'x-api-key',
            'cache_ttl_seconds': 0, 'timeout_seconds': 5}
        filters.insert(0, key_filter)
        filters.insert(1, {'filter': 'identity_header_guard', 'prefix': 'x-tenant-'})
        selected = active(models, original, api)
        # The unified renderer already denies inference for an unused API.
        # model_access rejects an empty allowlist at startup; retain auth and
        # the terminal 404 instead of constructing an invalid filter.
        if selected:
            # json_body rewrites during body pre-read, before model_access's
            # request hook. Check upstream IDs here; the unified renderer
            # separately rejects any unregistered client alias using the
            # original, body-derived X-Gateway-Model header.
            filters.insert(2, {'filter': 'model_access', 'mode': 'allowlist',
                               'models': sorted({model['model'] for model in selected}),
                               'group_metadata_key': 'x-tenant-group', 'max_body_bytes': 33554432})
        for i, item in enumerate(filters):
            if item['filter'] == 'token_count':
                filters.insert(i, _metering_filter())
                break
        if request_headers is not None and selected:
            load_balancer = next((i for i, item in enumerate(filters)
                                  if item['filter'] == 'load_balancer'), len(filters))
            filters.insert(load_balancer, request_headers)
        chain['filters'] = filters

    if settings is not None:
        openai = next((chain for chain in result['filter_chains'] if chain['name'] == 'openai'), None)
        if openai is None:
            raise ValueError('Switchyard requires an OpenAI gateway pipeline')
        result['filter_chains'].append(_switchyard_openai_chain(openai, original, settings, switchyard_catalog))

    allowed = [{'path': p} for p in DASHBOARD] + [{'path_prefix': p} for p in DASHBOARD_PREFIXES] + [
        {'path': '/v1/models', 'methods': ['GET']}] + [
        {'path': p, 'methods': ['POST']} for p in [*INFERENCE, '/v1/messages/count_tokens']]
    # The public listener routes protocol paths to different private chains so
    # OpenAI Bearer keys and Anthropic x-api-key credentials validate on
    # separate private chains. The public model catalog contains no secrets.
    public = [
        {'filter': 'request_id'},
        {'filter': 'static_response', 'status': 404, 'body': 'Route not available',
         'conditions': [{'unless': match} for match in allowed]},
        _public_rate_limit(),
        {'filter': 'static_response', 'status': 200, 'body': json.dumps(public_catalog),
         'headers': [{'name': 'Content-Type', 'value': 'application/json'}],
         'conditions': [{'when': {'path': '/v1/models', 'methods': ['GET']}}]},
        {'filter': 'router', 'routes':
            [{'path': p, 'cluster': 'authenticated-anthropic'}
             for p in ('/v1/messages', '/v1/messages/count_tokens')] +
            [{'path': p, 'cluster': 'authenticated-openai'}
             for p in ('/v1/responses', '/v1/chat/completions')] +
            [{'path': p, 'cluster': 'dashboard'} for p in DASHBOARD] +
            [{'path_prefix': p, 'cluster': 'dashboard'} for p in DASHBOARD_PREFIXES]},
        {'filter': 'load_balancer', 'clusters': [
            {'name': 'authenticated-openai', 'endpoints': ['127.0.0.1:18082']},
            {'name': 'authenticated-anthropic', 'endpoints': ['127.0.0.1:18083']},
            {'name': 'dashboard', 'endpoints': ['127.0.0.1:18085']}]}]
    dashboard = [
            {'filter': 'headers', 'request_remove': ['Authorization', 'X-Api-Key', 'X-MaaS-Username',
                'X-MaaS-Group', 'X-MaaS-Subscription', 'X-Tenant-Username', 'X-Tenant-Group',
                'X-Tenant-Subscription', 'X-Tenant-Model', 'X-Forwarded-User',
                'X-Forwarded-Groups', 'X-Forwarded-Real-User']},
            {'filter': 'router', 'routes': [{'path_prefix': '/', 'cluster': 'metering'}]},
            {'filter': 'load_balancer', 'clusters': [
                {'name': 'metering', 'endpoints': ['pricetag-metering:8080']}]}]
    result['filter_chains'] += [{'name': 'edge', 'filters': public}, {'name': 'dashboard', 'filters': dashboard}]
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--providers', type=Path, required=True,
                        help='rendered all-in-one Praxis config in JSON, with two native chains')
    parser.add_argument('--models', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.write_text(json.dumps(render(json.loads(args.providers.read_text()),
                                             json.loads(args.models.read_text())), indent=2) + '\n')


if __name__ == '__main__':
    main()
