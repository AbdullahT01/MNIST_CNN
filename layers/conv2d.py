import numpy as np

class Conv2D:
    def __init__(self, num_filters, filter_size = 3):
        self.num_filters = num_filters
        self.filter_size = filter_size
        # here we are making RANDOM filters, why random ?
        # well because we are going to make our cnn "learn" the right filters through backpropagation and gradient descent
        self.filters = np.random.randn(num_filters, 1, filter_size, filter_size) * 0.1 \
        
    def iterate_regions(self, image):
        h, w = image.shape

        for i in range (h - self.filter_size + 1):
            for j in range(w - self.filter_size + 1):
                # So if i = 0, j = 0 → region = image[0:3, 0:3] → the top-left 3×3 patch
                region = image[i: i + self.filter_size, j: j + self.filter_size]
                yield region, i, j

    # This function actually applies the convolution.
    def forward(self, input):
        self.last_input = input
        c, h, w = input.shape
        assert c == 1 # because we are using grayscale images
        output_dim = h - self.filter_size + 1
        output = np.zeros((self.num_filters, output_dim, output_dim))

        #Loop over each filter (we'll apply each filter to the entire image).
        for f in range(self.num_filters):
            for region, i, j in self.iterate_regions(input[0]):
                output[f, i, j] = np.sum(region * self.filters[f][0]) # this will basically place the convolved value, in the fth filter at the region i and j
        return output


    def backward(self, d_out, learning_rate):
        d_filters = np.zeros_like(self.filters)
        d_input = np.zeros_like(self.last_input)

        for f in range(self.num_filters):
            for region, i, j in self.iterate_regions(self.last_input[0]):
                d_filters[f][0] += d_out[f, i, j] * region
                d_input[0, i:i+self.filter_size, j:j+self.filter_size] += d_out[f, i, j] * self.filters[f][0]

        self.filters -= learning_rate * d_filters
        return d_input