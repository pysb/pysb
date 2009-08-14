import pysb
import reinteract
import gtk
import math


corner_radius = 10
site_radius = 5
sites_y_pos = 35
sites_y_spacing = 35


class MonomerWidget(gtk.DrawingArea):

    def __init__(self, monomer):
        gtk.DrawingArea.__init__(self)
        self.connect("expose_event", self.expose)
        self.set_size_request(100, sites_y_pos + sites_y_spacing * len(monomer.sites))
        self.monomer = monomer

    def expose(self, widget, event):
        self.draw()
        return False

    def draw(self):
        rect = self.get_allocation()
        context = self.context = self.window.cairo_create();

        context.set_line_width(1)

        # body
        (x1, x2) = (corner_radius + 0.5, rect.width - corner_radius - 0.5)
        (y1, y2) = (corner_radius + 0.5, rect.height - corner_radius - 0.5)
        angles = [v * math.pi/2.0 for v in range(0,4)]
        context.arc(x1, y1, corner_radius, angles[2], angles[3])
        context.arc(x2, y1, corner_radius, angles[3], angles[0])
        context.arc(x2, y2, corner_radius, angles[0], angles[1])
        context.arc(x1, y2, corner_radius, angles[1], angles[2])
        context.close_path()
        context.set_source_rgb(1, 1, 1)
        context.fill_preserve()
        context.set_source_rgb(0, 0, 0)
        context.stroke()

        # label
        context.set_font_size(14)
        context.move_to(corner_radius, context.font_extents()[0] + 3)
        context.show_text(self.monomer.name)
        context.move_to(0, context.font_extents()[0] + 8.5)
        context.rel_line_to(100, 0)
        context.stroke()

        # sites
        context.translate(corner_radius, sites_y_pos)
        for site in self.monomer.sites:
            self.draw_site(site)
            context.translate(0, sites_y_spacing)

    def draw_site(self, site):
        context = self.context
        context.set_source_rgb(0, 0, 0)
        context.arc(site_radius, 0, site_radius, 0, 2*math.pi)
        context.stroke()
        context.move_to(0, site_radius + context.font_extents()[0])
        context.show_text(site)
        context.new_path()





# mix-in with some pysb classes, but only if reinteract is available and running
mixin_pairs = [
    (pysb.Monomer, MonomerWidget),
    ]
def apply_mixins():
    for (base_class, widget_class) in mixin_pairs:
        if reinteract.custom_result.CustomResult not in base_class.__bases__:
            base_class.__bases__ += (reinteract.custom_result.CustomResult,)
            base_class.create_widget = lambda self: widget_class(self)
