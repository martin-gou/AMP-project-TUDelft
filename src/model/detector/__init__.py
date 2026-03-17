from .centerpoint import CenterPoint
from .cvfusion import CVFusion


def get_detector_class(config):
    model_type = str(config.get('type', 'CenterPoint')).lower()
    if model_type == 'cvfusion':
        return CVFusion
    if model_type == 'centerpoint':
        return CenterPoint
    raise ValueError(f'Unsupported detector type: {config.get("type")}')


def build_detector(config):
    return get_detector_class(config)(config)
