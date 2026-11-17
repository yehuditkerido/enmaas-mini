"""Validated, non-secret Switchyard settings for the PriceTag gateway."""
import copy
import json
import os
from pathlib import Path
import re
import stat


MODEL_NAME = re.compile(r'[A-Za-z0-9][A-Za-z0-9._:/-]{0,199}')


def _model_index(models):
    result = {}
    for model in models:
        identifier = model.get('id') if isinstance(model, dict) else None
        if not isinstance(identifier, str) or identifier in result:
            raise ValueError('models must contain unique model IDs')
        result[identifier] = model
    return result


def validate(settings, models):
    """Return a validated copy of an administrator-owned Switchyard selection.

    All referenced models must already be configured OpenAI-compatible models.
    The POC deliberately does not create credentials or discover providers.
    """
    if not isinstance(settings, dict):
        raise ValueError('Switchyard settings must be an object')
    allowed = {'alias', 'judge', 'weak', 'strong', 'fallback_model'}
    unknown = set(settings) - allowed
    if unknown:
        raise ValueError('unknown Switchyard setting: ' + sorted(unknown)[0])
    required = {'alias', 'judge', 'weak', 'strong'}
    missing = required - set(settings)
    if missing:
        raise ValueError('missing Switchyard setting: ' + sorted(missing)[0])
    alias = settings['alias']
    if not isinstance(alias, str) or not MODEL_NAME.fullmatch(alias):
        raise ValueError('Switchyard alias is invalid')
    indexed = _model_index(models)
    if alias in indexed:
        raise ValueError('Switchyard alias conflicts with an existing model ID')
    selected = {}
    for name in ('judge', 'weak', 'strong', 'fallback_model'):
        identifier = settings.get(name)
        if identifier is None:
            continue
        if not isinstance(identifier, str) or identifier not in indexed:
            raise ValueError('Switchyard ' + name + ' must name an existing model ID')
        model = indexed[identifier]
        if 'openai' not in model.get('apis', []):
            raise ValueError('Switchyard ' + name + ' must be OpenAI-compatible')
        selected[name] = copy.deepcopy(model)
    if ('fallback_model' in selected and
            selected['fallback_model']['id'] in {selected['weak']['id'], selected['strong']['id']}):
        raise ValueError('Switchyard fallback_model must differ from weak and strong')
    providers = {model['provider'] for model in selected.values()}
    if len(providers) != 1:
        raise ValueError('POC Switchyard models must use one existing provider')
    result = {'alias': alias, **selected}
    if 'fallback_model' in selected:
        result['fallback'] = True
    else:
        result['fallback'] = False
    return result


def normalize(settings, models):
    """Validate either administrator input or the renderer's saved state.

    ``load`` deliberately returns model metadata so callers do not need to
    repeat catalog lookups.  The gateway renderer accepts that saved state as
    well as the administrator's small ID-only document, but revalidates the
    IDs against the current model catalog before using it.
    """
    if not isinstance(settings, dict) or 'fallback' not in settings:
        return validate(settings, models)
    required = {'alias', 'judge', 'weak', 'strong', 'fallback'}
    allowed = required | {'fallback_model'}
    if set(settings) - allowed or required - set(settings):
        raise ValueError('invalid saved Switchyard settings')
    if not isinstance(settings['fallback'], bool):
        raise ValueError('invalid saved Switchyard fallback setting')
    raw = {'alias': settings['alias']}
    for name in ('judge', 'weak', 'strong', 'fallback_model'):
        value = settings.get(name)
        if value is None:
            continue
        if not isinstance(value, dict) or not isinstance(value.get('id'), str):
            raise ValueError('invalid saved Switchyard model selection')
        raw[name] = value['id']
    result = validate(raw, models)
    if result['fallback'] != settings['fallback']:
        raise ValueError('invalid saved Switchyard fallback setting')
    return result


def load(path, models):
    """Load and validate a JSON selection file.

    This is intentionally suitable for repository fixtures and other
    non-privileged callers.  Gateway administration paths must use
    :func:`load_private` instead.
    """
    path = Path(path)
    try:
        settings = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError('cannot read Switchyard settings') from error
    return validate(settings, models)


def load_private(path, models):
    """Read an administrator policy safely from a root-owned private file.

    The descriptor is opened without following a final symlink and inspected
    before its contents are parsed.  This prevents installation-time policy
    from being substituted between a pathname check and its use.
    """
    path = Path(path)
    if not path.is_absolute():
        raise ValueError('Switchyard config path must be absolute')
    flags = os.O_RDONLY | getattr(os, 'O_CLOEXEC', 0) | getattr(os, 'O_NOFOLLOW', 0)
    try:
        fd = os.open(path, flags)
        with os.fdopen(fd, 'rb') as source:
            info = os.fstat(source.fileno())
            if (not stat.S_ISREG(info.st_mode) or info.st_uid != 0 or
                    info.st_mode & 0o077):
                raise ValueError('Switchyard config must be a root-owned private regular file')
            settings = json.load(source)
    except ValueError:
        raise
    except (OSError, json.JSONDecodeError, UnicodeError) as error:
        raise ValueError('cannot read Switchyard settings') from error
    return validate(settings, models)
