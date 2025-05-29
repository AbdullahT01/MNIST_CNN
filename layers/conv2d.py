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

    def forward(self, input):
        self.last_input = input
        c, h, w = input.shape
        assert c == 1
        output_dim = h - self.filter_size + 1
        output = np.zeros((self.num_filters, output_dim, output_dim))

        for f in range(self.num_filters):
            for region, i, j in self.iterate_regions(input[0]):
                output[f, i, j] = np.sum(region * self.filters[f][0])
        return output
