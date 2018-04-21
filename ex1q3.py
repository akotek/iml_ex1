import numpy



def main():
    data = numpy.random.binomial(1, 0.25, (100000,1000))
    epsilon = [0.5, 0.25, 0.1, 0.01, 0.001]
    print(data)


if __name__ == '__main__':
    main()
