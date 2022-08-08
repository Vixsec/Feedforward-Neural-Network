import numpy as np



class Feedforward_Neural_Network( object ):
    """docstring for Feedforward_Neural_Network"""


    LEARNING_RATE         :float = None             # LEARNING_RATE set to None will use the average Error as the rate
    MAXIMUM_LEARNING_RATE :float = None             # instead of a specified float value and constraine it between the 
    MINIMUM_LEARNING_RATE :float = None             # minimun &or maximum rates if either rates are defined.....
    REPORT_PERCENT        :int   = None             # If set, progress updates are provided every INT%

    
    ACTIVATION_METHOD  :str = 'TANH';
    ACTIVATE :dict = { 
            'SIGMOID' : lambda DATA: 1 /( 1 +np.exp( -DATA ) ),
            'TANH'    : lambda DATA: np.tanh( DATA )
        };

    DERIVATIVE :dict = {
            'SIGMOID' : lambda DATA: DATA *( 1 -DATA ),
            'TANH'    : lambda DATA: 1 -( DATA**2 ),
        };

    DEFAULT_LRARTE_VALUES :dict = {                 # MINIMUM, LEARNING_RATE, MAXIMUM
            'TANH'    : [ 0.001,  None, 0.1 ],
            'SIGMOID' : [ 0.1,    None, 1.0 ]
        };




    def __init__( self, shape:list ):
        super( Feedforward_Neural_Network, self ).__init__( );
        structure = [ ( a, b ) for a, b in zip( shape[ 1: ], shape[ :-1 ] ) ];
        self.weights = [ np.random.standard_normal( s )/s[ 0 ]**0.5 for s in structure ];
        self.biases  = [ np.random.standard_normal( ( s, 1 ) ) for s in shape[ 1: ] ];

        self.setRates( default = True );



    # List requires [ minimum, setValue, maximum ], They can be of type float or None
    def setRates( self, rates:list=None, default:bool=False ) -> None:
        if( default ):
            self.MINIMUM_LEARNING_RATE, self.LEARNING_RATE, self.MAXIMUM_LEARNING_RATE = self.DEFAULT_LRARTE_VALUES[ self.ACTIVATION_METHOD ];

        else:
            self.MINIMUM_LEARNING_RATE  = rates[0];
            self.LEARNING_RATE          = rates[1];
            self.MAXIMUM_LEARNING_RATE  = rates[2];



    # Runs a single (forward & back) pass through the network
    def instance( self, inData:np.array, lbData:np.array ) -> None:
        datarray = self.feedforward( inData );
        self.backPropagation( datarray, lbData );



    # Pushes the input data through netwok!
    def feedforward( self, inData:list ) -> np.array:
        datarray = [ np.array( inData ) ];
        for weight, bias in zip( self.weights, self.biases ):
            datarray.append( self.ACTIVATE[ self.ACTIVATION_METHOD.upper( ) ]( np.matmul( weight, datarray[ -1 ] ) +bias ) );
        return datarray;



    # Backtracks through the network and applies the necessary corrections
    def backPropagation( self, inData:list, lbData:list ) -> None:
        ''' Takes in the datarray returned from the .feedforward(...) function and desired outcome label data array '''
        localError = lbData -inData[ -1 ];
        deltas = [ localError *self.DERIVATIVE[ self.ACTIVATION_METHOD.upper( ) ]( inData[ -1 ] ) ];

        LR = self.LEARNING_RATE if self.LEARNING_RATE else self.__learningConstraints( np.mean( np.abs( localError ) ) )

        for data, weight in zip( inData[ -2:0:-1 ], self.weights[ -1:0:-1 ] ):      
            localError = np.matmul( weight.T, deltas[ -1 ] );
            deltas.append( localError *self.DERIVATIVE[ self.ACTIVATION_METHOD.upper( ) ]( data ) );

        for data, delta, weight, bias in zip( inData[-2::-1], deltas, self.weights[ ::-1 ], self.biases[ ::-1 ] ):
            weight += np.matmul( delta, data.T ) *LR;
            bias   += np.sum( delta, axis=0, keepdims=True ) *LR;



    # Returns the output from the .feedforward()
    def predict( self, inData:np.array ) -> np.array:
        return self.feedforward( inData )[ -1 ]



    # Takes the training datasets (data & labels) and trains the network automatically 
    def arrayHandle( self, trainingData:list, trainingLabels:list, randomize:bool=True, epoc:float=1 ) -> None:
        if( len( trainingData ) != len( trainingLabels ) ): 
            raise "len( input data ) & len( label data ) don't match!";

        indexReference = np.random.permutation( len( trainingData ) ) if randomize else np.arange( len( trainingData ) )
        ITERATIONS = int( len( trainingData ) *epoc )

        for index in range( ITERATIONS ):
            self.instance( 
                trainingData[ indexReference[ index %len( indexReference ) ] ],
                trainingLabels[ indexReference[ index %len( indexReference ) ] ]
            )
            if( self.REPORT_PERCENT ):
                self.report( index, int( len( trainingData ) *epoc ), self.REPORT_PERCENT )



    # if the learning rate is not specified, return the constrained local_error... MIN<ERROR>MAX
    def __learningConstraints( self, ERROR:float ):
        if( self.MAXIMUM_LEARNING_RATE and ERROR > self.MAXIMUM_LEARNING_RATE ):
            return self.MAXIMUM_LEARNING_RATE;
        if( self.MINIMUM_LEARNING_RATE and ERROR < self.MINIMUM_LEARNING_RATE ):
            return self.MINIMUM_LEARNING_RATE;
        return ERROR;



    @staticmethod       # Alert the user every time the program completes self.REPORT_PERCENT%
    def report( CURRENT_ITERATION:int, TOTAL_ITERATIONS:int, PERCENT:int ) -> None:
        if CURRENT_ITERATION %int( np.ceil( ( TOTAL_ITERATIONS /100 ) *PERCENT ) ) == 0:
            print( f'Progress Update: { CURRENT_ITERATION /TOTAL_ITERATIONS :.1%}' )



    # Takes the testing datasets (data & labels) and reports network accuracy.
    def networkTest( self, testData:list, testLabels:list ) -> None :
        isEqual = lambda da, lb: ( da == lb ).all( );
        testedCorrect = 0;

        print( "Begining Tests:" );
        for data, label in zip( testData, testLabels ):
            result = self.feedforward( data )[ -1 ];
            if isEqual( np.transpose( result ).round( ), np.transpose( label ) ):
                testedCorrect +=1;

        testedCorrectPercent = float( testedCorrect ) /len( testData );
        print( '\nNetwork Tests Complete' );
        print( f'TESTED:{len( testData )}  ||  CORRECT:{testedCorrect}  ||  PCT:{testedCorrectPercent:.2%}\n' );



# -> LOAD DATAFRAME
DATAFRAME = np.load( "C:\\Dev\\_assets\\datasets\\mnist.npz" )

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