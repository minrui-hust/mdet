import mdet.data
import mdet.model
from mai.tools.eval import main, parse_args


r'''
Evaluate model on evaluation set.
Optionally save predict output and evaluation metric
'''

if __name__ == '__main__':
    main(parse_args())
