wget http://www.cs.toronto.edu/~kriz/cifar-10-binary.tar.gz
tar -xzvf cifar-10-binary.tar.gz
mv cifar-10-batches-bin/* ./
rm -r cifar-10-batches-bin
rm cifar-10-binary.tar.gz
