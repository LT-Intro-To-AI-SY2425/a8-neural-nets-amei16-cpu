from neural import *

print("<<<<<<<<<<<<<< XOR >>>>>>>>>>>>>>\n")
xor_training_data = [([0.9, 0.6, 0.8, 0.3, 0.1], [1]), ([0.8, 0.8, 0.4, 0.6, 0.4], [1]), ([0.7, 0.2, 0.4, 0.6, 0.3], [1]), ([0.5, 0.5, 0.8, 0.4, 0.8], [0]), ([0.3, 0.1, 0.6, 0.8, 0.8], [0]), ([0.6, 0.3, 0.4, 0.3, 0.6], [0])]

xorn = NeuralNet(5, 3, 1)
xorn.train(xor_training_data, iters= 100000, print_interval=10000 )
print(xorn.test_with_expected(xor_training_data))
print(xorn.evaluate([1.0, 1.0, 1.0, 0.1,  0.1]))
print(xorn.evaluate([0.5, 0.2, 0.1, 0.7,  0.7]))
print(xorn.evaluate([0.8, 0.3, 0.3, 0.3,  0.8]))
print(xorn.evaluate([0.8, 0.3, 0.3, 0.8,  0.3]))
print(xorn.evaluate([0.9, 0.8, 0.8, 0.3,  0.6]))


