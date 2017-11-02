#!/usr/bin/python

import SimpleHTTPServer
import SocketServer
import webbrowser


PORT = 9000

Handler = SimpleHTTPServer.SimpleHTTPRequestHandler
Handler.extensions_map.update({
    '.webapp': 'application/x-web-app-manifest+json',
})

httpd = SocketServer.TCPServer(("", PORT), Handler)

print "Serving at port", PORT
httpd.serve_forever()

# url = "file://Users/dionisio/PycharmProjects/pysb/pysb/tools/visualization/index.html"
# webbrowser.open(url, new=2)
