import numpy as np

class MaxPool2:
    def iterate_regions(self, image):
        h, w = image.shape
        new_h = h // 2
        new_w = w // 2
        for i in range(new_h):
            for j in range(new_w):
                region = image[(i*2):(i*2+2), (j*2):(j*2+2)]
                yield region, i, j

    def forward(self, input):
        self.last_input = input  # ← Add this line
        c, h, w = input.shape
        output = np.zeros((c, h // 2, w // 2))
        for ch in range(c):
            for region, i, j in self.iterate_regions(input[ch]):
                output[ch, i, j] = np.max(region)
        return output

    
    def backward(self, d_out):
            input = self.last_input
            c, h, w = input.shape
            d_input = np.zeros_like(input)

            for ch in range(c):
                for i in range(0, h, 2):
                    for j in range(0, w, 2):
                        patch = input[ch, i:i+2, j:j+2]
                        max_val = np.max(patch)
                        for ii in range(2):
                            for jj in range(2):
                                if patch[ii, jj] == max_val:
                                    d_input[ch, i+ii, j+jj] = d_out[ch, i//2, j//2]
            return d_input