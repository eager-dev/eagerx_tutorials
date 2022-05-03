
import eagerx
import eagerx.converters  # Registers space converters
from eagerx.utils.utils import Msg
from std_msgs.msg import Float32, Float32MultiArray


class MovingAverageFilter(eagerx.Node):
    @staticmethod
    @eagerx.register.spec("ExampleNode", eagerx.Node)
    def spec(
        spec,
        name: str,
        rate: float,
        n: int,
    ):
        """
        MovingAverage filter
        :param spec: Not provided by user.
        :param name: Node name
        :param rate: Rate at which callback is called.
        :param n: Window size of the moving average
        :return:
        """
        # Performs all the steps to fill-in the params with registered info about all functions.
        spec.initialize(MovingAverageFilter)

        # Modify default node params
        spec.config.name = name
        spec.config.rate = rate
        spec.config.process = eagerx.process.ENVIRONMENT
        spec.config.inputs = ["signal"]
        spec.config.outputs = ["filtered"]
        
        # Custom node params
        # START ASSIGNMENT 1.1

        # START ASSIGNMENT 1.1
        
        # Add space converters
        spec.inputs.signal.space_converter = eagerx.SpaceConverter.make("Space_Float32", -3, 3, dtype="float32")
        spec.outputs.filtered.space_converter = eagerx.SpaceConverter.make("Space_Float32MultiArray", [-3], [3], dtype="float32")
    
    # START ASSIGNMENT 1.2
    def initialize(self):
        pass
    # END ASSIGNMENT 1.2
    
    @eagerx.register.states()
    def reset(self):
        # START ASSIGNMENT 1.3
        pass
        # END ASSIGNMENT 1.3

    @eagerx.register.inputs(signal=Float32)
    @eagerx.register.outputs(filtered=Float32MultiArray)
    def callback(self, t_n: float, signal: Msg):
        data = signal.msgs[-1].data
        
        # START ASSIGNMENT 1.4
        filtered_data = data
        # END ASSIGNMENT 1.4
        
        return dict(filtered=Float32MultiArray(data=[filtered_data]))
