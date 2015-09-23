#import bias_model_tan_noFS_spider.py 
import sys
import argparse
import os
def main(argv):
	args = parse_args()
	input_train_file = args.input_train_file
	input_test_file  = args.input_test_file
	output_file      = args.output_file
	delim            = args.delim
    
	print "running"

	call_str="bias_model_noFS.py "+input_train_file+" "+input_test_file+" "+output_file+" --delim=\""+delim+"\""
	#print call_str
	for i in range(1):
		os.system("python "+call_str+" "+str(i))

def parse_args():
    parser = argparse.ArgumentParser(description="Classifier Caller")

    parser.add_argument("input_train_file",     type=str, help="the input train file")
    parser.add_argument("input_test_file",      type=str, help="the input test file")
    parser.add_argument("output_file",          type=str, help="the output file with predictions")
    parser.add_argument("--delim", default=',', type=str, help="the delimiter in the train, test, and predictions file")

    args = parser.parse_args()
    # Is this return necessary in Python?
    return args

if __name__=="__main__":
    main(sys.argv[1:])