# Feedforward-Neural-Network
An easily resizeable Feedforward neural network, that fits to any desired shape


# How to use:

    Pre-init setup: Specify the desired ACTIVATION_METHOD( TANH by default ).
            -> Feedforward_Neural_Network.ACTIVATION_METHOD = 'SIGMOID';
    if it gets set after the network is initialized, use the .setRates( list_of_rates, default=True ) function to change the rates from the TANH default options.

    Expectd input format for data = numpy.zeros( ( N, 1 ) )

    Initialize the network with an array of weights where each index holds the number_of_nodes for that layer
            -> FNN = Feedforward_Neural_network( shape=[ 2, 3, 1 ] ) -> [ 2(inputs), 3(hiddenlayer), 1(output)
            -> shape=[ 784, 16, 16, 10 ] -> [ 784(Inputs), 16(HL1), 16(HL2), 10(Outputs) ]
    there is no limit to the number of nodes or indexes in shape[].

    To pass data through the network, you have three options
        (1) Access the .feedforward/.backpropagation functions indavidually, both requiring single instances of data.
        (2) Use the .instance( data, label ) function, also takes single instances of data, but runs runs option (1) inside of it.
        (3) Use the .arrayHandle( training_data, training_label ) function, which will run the entire array sets, 
            and provide randomization(bool) & epoc(int) options.

    Test the network with .networkTest( testData, testLabels ), which will test the entire array set and print the accuracy of the network.
            -> print( f'TESTED:{len( testData )}  ||  CORRECT:{testedCorrect}  ||  PCT:{testedCorrectPercent:.2%}\n' );



# Class Variables:

    Learning Rate : FLOAT
        Hard coding the LEARNING_RATE will use the provided 'float' value by default.
        Setting the LEARNING_RATE to None will use the average Error as the learning rate instead of a specific float value and 
        constraine it between the minimun_learning_rate &or maximum_learning_rate if either rates are defined.....

    REPORT_PERCENT : INT
        If set, progress/completion updates are provided every N%
        If None, Reports get ignored

    ACTIVATION_METHOD : STR.upper()
        The ACTIVATION_METHOD tells the network what activation and derivative functions to use as well as 
        providing some default LEARNING_RATE(s) for each method.
        
        Currently works with TANH(default) & SIGMOID, but since activation & derivative functions are nested in dictionaries,
        its very easy to add new functions at any time.



# Adding new Activation & derivative function
    
    Set the .ACTIVATION_METHOD to a new id: ACTIVATION_METHOD = 'NEW_TITLE_ID'

    To add a new lambda functions to the dictionary:
                Feedforward_Neural_Network.ACTIVATE[ ACTIVATION_METHOD ] = lambda DATA: doStuff toDATA;
                Feedforward_Neural_Network.DERIVATIVE[ ACTIVATION_METHOD ] = lambda DATA: doStuff toDATA;

    To add a more indepth functions, pass a function to the lambda function:
                Create Function::  def activate_function( data ): return sum( data );
                Feedforward_Neural_Network.ACTIVATE[ ACTIVATION_METHOD ] = lambda DATA: activate_function( data );
                Create Function::  def derivative_function( data ): return sum( data );
                Feedforward_Neural_Network.DERIVATIVE[ ACTIVATION_METHOD ] = lambda DATA: derivative_function( data );



# XOR Problem Example:

    # -> CREATE DATASET & LABELS
        data   = [ [ [0], [0] ], [ [0], [1] ], [ [1], [0] ], [ [1], [1] ] ];
        labels = [      [-1],         [1],         [1],          [-1] ];                # Use TANH as method for -1 values

    # -> CREATE NETWORK
        FNN = Feedforward_Neural_Network( shape = [ 2, 3, 1 ] );
        
    # -> TRAIN Network                                              # Iterations = epoc *len( dataset )
        FNN.arrayHandle( data, labels, randomize=True, epoc=2000 )
        
    # -> TEST NETWORK
        FNN.networkTest( data, labels )



# MNIST Dataset Example:

    # -> LOAD DATAFRAME
        DATAFRAME = np.load( "mnist.npz" )

    # -> Pre-Set the activation method to sigmoid because the mnist dataset labels run between 0-1.
        Feedforward_Neural_Network.ACTIVATION_METHOD = 'SIGMOID'

    # -> CREATE THE NETWORK               shape=[ 784, 16, 16, 10 ]
        FNN = Feedforward_Neural_Network( shape=[ len( DATAFRAME['training_images'][ 0 ] ), 16, 16, 10 ] )

    # -> Set report percent to 10% for active updates
        Feedforward_Neural_Network.REPORT_PERCENT = 10

    # -> TRAIN THE NETWORK
        FNN.arrayHandle( DATAFRAME['training_images'], DATAFRAME['training_labels'], randomize=False, epoc=1 )      # iterations = epoc *len( data )

    # -> TEST THE NETWORK
        FNN.networkTest( DATAFRAME['test_images'], DATAFRAME['test_labels'] )
