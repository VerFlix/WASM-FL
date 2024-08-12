cd fann/src
$CC -std=c++14 -c -o fann.o fann.c
ar rcs libfann.a fann.o
cd ../..
cd src/
$CC -std=c++14 -c -o *.o *.c
ar rcs *.a *.o