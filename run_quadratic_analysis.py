import sys
import argparse
from QuadraticAnalysisToolkit import Grid, QuadraticAnalysis, EnsembleAnalysis

parser = argparse.ArgumentParser()
parser.add_argument('csvfile', type=str,
                    help='Name of the input csv file containing (x, y, value) data series.')
parser.add_argument('-d', '--csvdelimiter', type=str, default=',',
                    help='Delimiter for csv file (default is comma).')
parser.add_argument('-h', '--skipheader', type=int, default=1,
                    help='Number of header lines in csv to skip (default is 1).')
parser.add_argument('-lo', '--lo', type=float, nargs='+', required=True,
                    help='Lower bounds of rectangular input uncertainty region.')
parser.add_argument('-hi', '--hi', type=float, nargs='+', required=True,
                    help='Upper bounds of rectangular input uncertainty region.')
parser.add_argument('-o','--output', type=str, default='quad.log',
                    help='Name of output file for writing the results (default quad.log).')
parser.add_argument('-v','--verbose', action='store_true',
                    help='If supplied, perform analysis with verbose status printing.')
parser.add_argument('-nens', '--numensemble', type=int,
                    help='(Optional) Repeat the quadratic analysis to construct an ensemble, with each quadratic fit in the ensemble considering only numensemble points in the csv. The size of the ensemble will be the number of unique combinations of the points in the csv, selected numensemble at a time.')
args = parser.parse_args()
        
if __name__=='__main__':
    # First do sanity checking on inputs
    if len(args.lo) != len(args.hi):
        sys.exit('ERROR: -lo and -hi boundary lists should be the same length')

    # Set dimensionality
    ndim = len(args.lo)
        
    # Open the csv file in a Grid object
    g = Grid()
    g.initFromCSV(args.csvfile,
                  delimiter=args.csvdelimiter,
                  skip_header=args.skipheader)

    # Make sure grid is the same dimensionality as lo and hi
    if ndim != g.dm:
        sys.exit('ERROR: Boundary lists should be the same length as the csv point dimensionality.')

    # Check to make sure the grid is large enough for quadratic analysis
    npoints_required = ndim(ndim-1)/2 + 2*ndim + 1
    if len(g.points) < npoints_required:
        sys.exit('ERROR: Cannot perform quadratic analysis in {} dimensions with fewer than {} points.'.format(ndim, npoints_required))
    
    # Detect whether we should do an ensemble
    if (args.numensemble and
        args.numensemble >= npoints_required and
        args.numensemble < len(g.points)):
        ea = EnsembleAnalysis(g, args.lo, args.hi, args.numensemble,
                              args.output, args.verbose)
    # Otherwise do 1 quadratic analysis using all grid points 
    else:
        qa = QuadraticAnalysis(g, args.lo, args.hi,
                               args.output, args.verbose)
