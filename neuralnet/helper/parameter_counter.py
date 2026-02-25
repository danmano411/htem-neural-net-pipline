INPUT = 1386
OUTPUT = 1

def count_params(layer_sizes):
    """
    layer_sizes: list like [input, h1, h2, ..., output]
    """
    return sum(
        layer_sizes[i] * layer_sizes[i+1] + layer_sizes[i+1]
        for i in range(len(layer_sizes) - 1)
    )

hidden_layer_size_grid = [
    # [2, 4, 8, 16, 8], #3115
    # [2, 8, 16, 32, 8], # 3759
    [2, 4, 16, 32, 16], # 3955 ***********
    # [2, 8, 16, 32, 16], # 4031 
    # [2, 4, 8, 16, 32, 16], # 4059
    # [2, 8, 16, 32, 16, 8], # 4159
    # [2, 4, 8, 16, 16, 32, 16], # 4331
    # [2, 4, 8, 16, 32, 32, 16], # 5115
    [2, 4, 8, 32, 32, 16], # 4715 ***********
    [4, 8, 16, 8], # 5877 *******
    [4, 8, 16, 32, 16], # 6821 *****
    # [4, 8, 16, 32, 16, 8], # 6949
    [4, 8, 16, 32, 32, 8], # 7605 ******
    # [4, 8, 16, 64, 16], # 7877
]

for layers in hidden_layer_size_grid:
    print(count_params([INPUT, *layers, OUTPUT]))