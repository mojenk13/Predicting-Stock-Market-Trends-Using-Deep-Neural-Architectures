from graphviz import Digraph

dot = Digraph()

dot.node('A', 'Input (15×5)')
dot.node('B', 'VAE Encoder')
dot.node('C', 'Latent Space (64D)')
dot.node('D', 'RBF Kernel')
dot.node('E', 'Linear Output')
dot.node('F', 'Prediction')

dot.edges(['AB', 'BC', 'CD', 'DE', 'EF'])

dot.render('vae_elm_flow', format='png', view=True)
