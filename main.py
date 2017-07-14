"""
Primary execution.

@author Sam O | <samuel.ordonia@gmail.com>
"""
import argparse
from scripts import iris_analysis


"""CLARGS"""
parser = argparse.ArgumentParser(
    description='Classifications Basics:  Tensorflow Exercises\n\
    Contents:\n\
        1) Iris flower linear classification\n\
        2) Word2Vec exercise',
    formatter_class=argparse.RawDescriptionHelpFormatter,
    epilog='For further questions, please consult the README.'
)

# Add CLARGS
parser.add_argument(
    '-i',
    '--iris-analysis',
    action='store_true',
    help='Iris flower classification model to predict flower species\n\
        based on sepal/petal geometry.'
)
parser.add_argument(
    '-w',
    '--word-2-vec',
    action='store_true',
    help='Vector Representations of Words.'
)

if __name__ == '__main__':
    args = parser.parse_args()

    if args.iris_analysis:
        print('Running basic classification model on iris flowers')
        iris_analysis()
