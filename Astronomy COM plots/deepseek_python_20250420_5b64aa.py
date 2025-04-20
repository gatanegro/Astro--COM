# Modified map_to_octave function with HQS-LZ scaling
def map_to_octave(value, layer):
    scaled_value = value * 1.23498 * (1 - 0.235*np.sin(layer/4))
    angle = (scaled_value / 9) * 2 * np.pi
    radius = (layer + 1) * (1 + 0.235*np.cos(layer/2))
    x = np.cos(angle) * radius
    y = np.sin(angle) * radius
    return x, y