PEN = {
	'name' : 'PEN',
	'bidir' : False,
	'clip_val' : 10,
	'drop_prob' : 0.5,
	'n_epochs_hold' : 100,
	'n_layers' : 1,
	'learning_rate' : [0.001],
	'weight_decay' : 0.0001,
	'n_residual_layers' : 2,
	'diag' : 'Architecure chosen is PEN',
	'save_file' : 'results_pen.txt'
}

Architecture={
    'PEN': PEN
}
# Choose what architecure you want here:
arch = Architecture['PEN']

# This will set the values according to that architecture
bidir = arch['bidir']
clip_val = arch['clip_val']
drop_prob = arch['drop_prob']
n_epochs_hold = arch['n_epochs_hold']
n_layers = arch['n_layers']
learning_rate = arch['learning_rate']
weight_decay = arch['weight_decay']
# n_highway_layers = arch['n_highway_layers']
# n_residual_layers = arch['n_residual_layers']

# These are for diagnostics
diag = arch['diag']
save_file = arch['save_file']

n_classes = 6
n_input = 9
n_hidden = 64
batch_size = 64
n_epochs = 50