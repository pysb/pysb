import Pysb
import gtk
import math


class MonomerWidget(gtk.DrawingArea):

    def __init__(self, monomer):
        gtk.DrawingArea.__init__(self)
        self.connect("expose_event", self.expose)
        self.set_size_request(100, 30 + 30 * len(monomer.sites))
        self.monomer = monomer

    def expose(self, widget, event):
        self.draw()
        return False

    def draw(self):
        rect = self.get_allocation()
        context = self.context = self.window.cairo_create();

        x = rect.width / 2
        y = rect.height / 2

        radius = min(rect.width / 2, rect.height / 2) - 5

        # body
        context.arc(x, y, radius, 0, 2 * math.pi)
        context.set_source_rgb(1, 1, 1)
        context.fill_preserve()
        context.set_source_rgb(0, 0, 0)
        context.stroke()

        # label
        context.set_font_size(14)
        context.move_to(3, context.font_extents()[0] + 3)
        context.show_text(self.monomer.name)

        # sites
        context.translate(10, 30)
        for site in self.monomer.sites:
            self.draw_site(site)
            context.translate(0, 20)

    def draw_site(self, site):
        context = self.context
        context.set_source_rgb(0, 0, 0)
        context.move_to(5, 5)
        context.show_text(site)
        context.new_sub_path()
        context.arc(0, 0, 5, 0, 2*math.pi)
        context.stroke()



def apply_mixin(base_class, widget_class):
    if reinteract.custom_result.CustomResult not in base_class.__bases__:
        base_class.__bases__ += (reinteract.custom_result.CustomResult,)
        base_class.create_widget = lambda self: widget_class(self)


# mix-in with some Pysb classes, but only if reinteract is available and running
class_pairs = [
    (Pysb.Monomer, MonomerWidget),
    ]
try:
    import reinteract         # fails if reinteract not installed
    reinteract.custom_result  # fails if this code is run outside of the reinteract shell
except (ImportError, AttributeError):
    pass                      # silently skip applying the mixin below
else:
    for pair in class_pairs:
        apply_mixin(*pair)
