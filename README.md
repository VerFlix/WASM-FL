This repository is part of my bachelor thesis. The WASM FL framework can be seen as a research project. Feel free to participate or engage in givin ideas via issues. 
I will add some features in the future and hope to be able to write my student project during the master or my master thesis about optimizing or implementing aspects of the framework.


Since the main objective of this work has been on the framework itself, we have not yet created an easy-to-use compilation method, but one can be added in the future. For running the framework, the WASM server and client binaries can be found in the framework directory. As the initial global model, the model in networks/global_model.net is used. When needed, one of the sample networks, created when running the create_networks.wasm binary in the networks folder can be used. When running the server, the number of clients needed to perform training before aggregating the new model is passed as the argument. The server will listen on port 8088 for the clients to connect. When running the client, the IP address of the server is passed as the argument, and the data in data/local.data is used for training. The data needs to be in the FANNs data format. This format is a plain text file with a specific structure. The first line consists of the number of training samples, the number of input neurons, and the number of output neurons. The following lines contain the training samples, each sample having two lines. The first line contains the input value set, and the second line contains the corresponding output value set. Both value sets are listed using whitespaces.

 In order to compile custom FL algorithms, first, the source code must be compiled into a static library. This can be done by running the shell script in the include folder. The script compiles the FANN library and our library into static libraries. Because it is not always easy to compile code to WASM binaries, one way of doing it is shown in the Readme file of the library. This approach uses the clang compiler and the WASI Software Development Kit ( SDK ). It includes the header files, specifies the source directories, and links against the static libraries. Also the WASI SDK GitHub is referenced for installing the WASI SDK.


install WASI SDK: https://github.com/WebAssembly/wasi-sdk

compile the .c files to .a static libraries. You can use the setup file for that. Make sure $CC is your clang compiler.

compile your .c pr .cpp to .wasm using your clang compiler with -L/path/to/library/files -IL/path/to/include/files and -l[library] arguments.