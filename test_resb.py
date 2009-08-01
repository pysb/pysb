from Pysb import *
from resb import *
import gtk

Model('test')
m = Monomer( 'EGFR', ['l','r','Y1068','Y1148'], { 'Y1068': ['U','P'], 'Y1148': ['U','P'] } )
window = gtk.Window()
mw = MonomerWidget(m)
window.add(mw)
window.connect("destroy", gtk.main_quit)
window.connect("key-press-event", gtk.main_quit)
window.show_all()
gtk.main()
