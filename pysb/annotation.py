from pysb.core import SelfExporter, Component

class Annotation(object):

    """A lightweight annotation mechanism for model components and other
    elements"""

    def __init__(self, subject, object_, predicate="is"):
        self.subject = subject
        self.object = object_
        self.predicate = predicate
        # if SelfExporter is in use, add the annotation to the model
        if SelfExporter.do_export:
            SelfExporter.default_model.add_annotation(self)

    def __repr__(self):
        if isinstance(self.subject, Component):
            subject = self.subject.name
        else:
            subject = self.subject
        return "%s(%s, %s, %s)" % (self.__class__.__name__, subject,
                                   repr(self.object), repr(self.predicate))
