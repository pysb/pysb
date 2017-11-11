if (window['cytoscape'] === undefined) {
    console.log('starting loading');
    requirejs.config({

        paths: {
            'cytoscape': 'https://cdnjs.cloudflare.com/ajax/libs/cytoscape/2.7.23/cytoscape.min',
            'cytoscape-qtip': 'https://cdn.rawgit.com/cytoscape/cytoscape.js-qtip/2.7.0/cytoscape-qtip',
			'jquery': 'https://cdnjs.cloudflare.com/ajax/libs/jquery/2.2.4/jquery.min',
            'qtip2': 'https://cdnjs.cloudflare.com/ajax/libs/qtip2/2.2.0/basic/jquery.qtip.min'
        }
    });
    window.$ = window.jQuery = require('jquery');
    require(['cytoscape', 'cytoscape-qtip', 'jquery', 'qtip2'], function (cytoscape, cyqtip, jquery, registerqtip) {
        console.log('Loading Cytoscape.js Module...');
        cyqtip(cytoscape, jquery);
        window['cytoscape'] = cytoscape;

        var event = document.createEvent("HTMLEvents");
        event.initEvent("load_cytoscape", true, false);
        window.dispatchEvent(event);
    });
}