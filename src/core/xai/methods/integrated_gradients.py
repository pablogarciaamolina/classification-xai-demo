from captum.attr import IntegratedGradients

class IGExplainer():
    def __init__(self, model):
        self.model = model
        self.ig = IntegratedGradients(model)
 
    def explain(self, input_tensor, class_idx):
        attributions = self.ig.attribute(input_tensor, target=class_idx, n_steps=50)
        g = attributions.sum(dim=1).squeeze().cpu().detach()
        g = g - g.min()
        g /= (g.max() + 1e-8)
        return g