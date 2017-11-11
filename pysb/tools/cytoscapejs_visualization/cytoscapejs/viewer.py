import uuid
import json
import os

# ######## Default Values ###########

# Define default widget size
DEF_HEIGHT = 700
DEF_WIDTH = 100 # Same as cell width of Jupyter

DEF_BACKGROUND_COLOR = '#FFFFFF'

HTML_TEMPLATE_FILE = 'template.html'
STYLE_FILE = 'default_style.json'


DEF_LAYOUT = 'preset'
DEF_STYLE = 'default2'

PRESET_LAYOUTS = {
    'Preset': 'preset',
    'Circle': 'circle',
    'Concentric': 'concentric',
    'Breadthfirst': 'breadthfirst',
    'Spring': 'cose',
    'Grid': 'grid'
}


# Init styles
style_file = open(os.path.abspath(os.path.dirname(__file__)) + '/' + STYLE_FILE, 'r')
style_list = json.load(style_file)
STYLES = {}
for style in style_list:
    STYLES[style['title']] = style['style']


def render(network,
           style=DEF_STYLE,
           layout_algorithm=DEF_LAYOUT,
           background=DEF_BACKGROUND_COLOR,
           height=DEF_HEIGHT,
           width=DEF_WIDTH):
    """Render network data with embedded Cytoscape.js widget.

    :param network: dict (required)
        The network data should be in Cytoscape.js JSON format.
    :param style: str or dict
        If str, pick one of the preset style. [default: 'default']
        If dict, it should be Cytoscape.js style CSS object
    :param layout_algorithm: str
        Name of Cytoscape.js layout algorithm
    :param background: str
        Background in CSS format
    :param height: int
        Height of the widget.
    :param width: int
        Width of the widget.
    """

    from jinja2 import Template
    from IPython.core.display import display, HTML

    # Load style file if none available
    if isinstance(style, str):
        # Specified by name
        style = STYLES[style]

    if network is None:
        raise Exception('a network dict must be provided')

    path = os.path.abspath(os.path.dirname(__file__)) + '/' + HTML_TEMPLATE_FILE
    template = Template(open(path).read())
    cyjs_widget = template.render(
        datos=json.dumps(network),
        background=background,
        uuid="cy" + str(uuid.uuid4()),
        fitbutton = "fit" + str(uuid.uuid4()),
        playid="play" + str(uuid.uuid4()),
        resetid="reset" + str(uuid.uuid4()),
        rangeid="range" + str(uuid.uuid4()),
        textid="text" + str(uuid.uuid4()),
        widget_width=str(width),
        widget_height=str(height),
        layout=layout_algorithm,
        style_json=json.dumps(style)
    )

    display(HTML(cyjs_widget))


# List of available layout algorithms
def get_layouts():
    return PRESET_LAYOUTS


def get_style_names():
    return list(STYLES.keys())


def get_style(name):
    if name in STYLES.keys():
        return STYLES[name]
    else:
        raise ValueError('Style does not exist: ' + name)