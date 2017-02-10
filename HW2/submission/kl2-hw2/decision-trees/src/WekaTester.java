package cs446.homework2;
import java.io.File;
import java.io.FileReader;

import weka.classifiers.Evaluation;
import weka.core.Instances;
import weka.core.Instance;
import weka.core.Attribute;
import weka.core.FastVector;
import cs446.weka.classifiers.trees.Id3;
import cs446.weka.classifiers.trees.SGD;
import java.util.Random;
public class WekaTester {
    public static void main(String[] args) throws Exception {

	// Load the data
    Instances data[]= new Instances[5];
	for(int i=0;i<5;i++){
		data[i]=new Instances(new FileReader(new File(args[i])));
		data[i].setClassIndex(data[i].numAttributes()-1);
	}

	// dividing test sets into 5-fold test-train pairs
	Instances trains[]=new Instances[5];
	Instances tests[]=new Instances[5];
	for(int i=0;i<5;i++){
		trains[i]=new Instances(data[i]);
		trains[i].delete();
		for(int j=0;j<5;j++){
			if(j!=i){
				for(int k=0;k<data[j].numInstances();k++){
					trains[i].add(data[j].instance(k));
				}
			}
		}	
		tests[i]=new Instances(data[i]);	
	}


	//Create 5 SGDs classifiers
 	/***********************************************************************************************************************************************/
	SGD sgds[]=new SGD[5];
	for(int i=0;i<5;i++){
		sgds[i]=new SGD(trains[0].instance(0).numAttributes()-1);
		// tunning the parameters before trainning
		// sgds[i].parametersTuning(trains,tests);
	}
	// Train the SGDs
	for(int i=0;i<5;i++){
		sgds[i].train(trains[i]);
	}
	// Test them on the cross validation set
	for(int i=0;i<5;i++){
		System.out.println("SGD,fold"+(i+1)+"\n");
		sgds[i].test(tests[i]);
	}
	/***********************************************************************************************************************************************/

	// Create 5 new ID3 classifiers. This is the modified one where you can
	//set the depth of the tree.

	// Id3 with unlimited depth
	Id3 classifiers[]=new Id3[5];
	for(int i=0;i<5;i++){
		classifiers[i]=new Id3();
		classifiers[i].setMaxDepth(-1);
		classifiers[i].buildClassifier(trains[i]);		
	}

	// Evaluate on the test set
	Evaluation evaluations[]= new Evaluation[5];
	for(int i=0;i<5;i++){
		evaluations[i]=new Evaluation(tests[i]);
		evaluations[i].evaluateModel(classifiers[i],tests[i]);
	}

	//print out the results
	for(int i=0;i<5;i++){
		System.out.println("ID3 with unlimited depth, fold"+(i+1)+"\n");
		System.out.println(classifiers[i]);
		System.out.println();
		System.out.println(evaluations[i].toSummaryString());
	}

	//Id3 with depth of 4
	Id3 classifiers1[]=new Id3[5];
	for(int i=0;i<5;i++){
		classifiers1[i]=new Id3();
		classifiers1[i].setMaxDepth(4);
		classifiers1[i].buildClassifier(trains[i]);		
	}

	// Evaluate on the test set
	Evaluation evaluations1[]= new Evaluation[5];
	for(int i=0;i<5;i++){
		evaluations1[i]=new Evaluation(tests[i]);
		evaluations1[i].evaluateModel(classifiers1[i],tests[i]);
	}

	//print out the results
	for(int i=0;i<5;i++){
		System.out.println("ID3 with depth of 4,fold"+(i+1)+"\n");
		System.out.println(classifiers1[i]);
		System.out.println();
		System.out.println(evaluations1[i].toSummaryString());
	}

	//Id3 with depth of 8
	Id3 classifiers2[]=new Id3[5];
	for(int i=0;i<5;i++){
		classifiers2[i]=new Id3();
		classifiers2[i].setMaxDepth(8);
		classifiers2[i].buildClassifier(trains[i]);		
	}

	// Evaluate on the test set
	Evaluation evaluations2[]= new Evaluation[5];
	for(int i=0;i<5;i++){
		evaluations2[i]=new Evaluation(tests[i]);
		evaluations2[i].evaluateModel(classifiers2[i],tests[i]);
	}

	//print out the results
	for(int i=0;i<5;i++){
		System.out.println("ID3 with depth of 8,fold"+(i+1)+"\n");
		System.out.println(classifiers2[i]);
		System.out.println();
		System.out.println(evaluations2[i].toSummaryString());
	}

    // decision stumps
	/***********************************************************************************************************************************************/
	System.out.println("SGD with decision stumps as features"+"\n");
	//100 stumps for one fold, 5 folds have 500 stumps in total
	Id3 stumps[][]=new Id3[5][100];
	for(int i=0;i<5;i++){
		for(int j=0;j<100;j++){
			//initialize the decision tree
			stumps[i][j]=new Id3();
			//set each of its depth to 4
			stumps[i][j].setMaxDepth(4);
			//initialize the set, because this class does not have defalut empty constructor, I have to use copy constructor then clean it afterward
			Instances subset=new Instances(trains[0]);
			//delete everything in the set
			subset.delete();
			// seed for random number generator
			long seed=i*10+j;
			//shuffle the trainning set
			trains[i].randomize(new Random(seed));
			//get half of the data
			for(int k=0;k<trains[i].numInstances()/2;k++){
				subset.add(trains[i].instance(k));
			}
			//train a stump
			stumps[i][j].buildClassifier(subset);
		}
	}
	// using the stumps to generate new feature sets for the training data
	Instances new_trains[]=new Instances[5];
	for(int i=0;i<5;i++){
		//initialize the attribute information array(names of the attribute)
		FastVector attInfo=new FastVector(101);
		//give them names from "0" to "99"
		for(int j=0;j<100;j++){
			attInfo.addElement(new Attribute(Integer.toString(j)));
		}
		// add the label to the end of the attribute information list
		attInfo.addElement(new Attribute("label",(FastVector)null));
		// initialize an empty set
		new_trains[i]=new Instances("fold"+i,attInfo,70);
		// adding generated features (100 dimension) into the new set
		for(int j=0;j<trains[i].numInstances();j++){
			Instance temp=new Instance(101);
			temp.setDataset(new_trains[i]);
			for(int k=0;k<100;k++){
				//generate new data set and add it to the new instance
				temp.setValue(k,stumps[i][k].classifyInstance(trains[i].instance(j)));
			}
			temp.setValue(100,trains[i].instance(j).stringValue(trains[i].instance(j).numAttributes()-1));
			new_trains[i].add(temp);
		}
		//set Class index to the last attribute(the label)
		new_trains[i].setClassIndex(new_trains[i].numAttributes()-1);
	}
	// using the stumps to generate new feature sets for the testing data
	Instances new_tests[]=new Instances[5];
	for(int i=0;i<5;i++){
		FastVector attInfo=new FastVector(101);
		for(int j=0;j<100;j++){
			attInfo.addElement(new Attribute(Integer.toString(j)));
		}
		attInfo.addElement(new Attribute("label",(FastVector)null));
		new_tests[i]=new Instances("fold"+i,attInfo,70);
		for(int j=0;j<tests[i].numInstances();j++){
			Instance temp=new Instance(101);
			temp.setDataset(new_tests[i]);
			for(int k=0;k<100;k++){
				temp.setValue(k,stumps[i][k].classifyInstance(tests[i].instance(j)));
			}
			temp.setValue(100,tests[i].instance(j).stringValue(trains[i].instance(j).numAttributes()-1));
			new_tests[i].add(temp);
		}
		new_tests[i].setClassIndex(new_tests[i].numAttributes()-1);
	}

	
	//Create 5 SGDs classifiers
	SGD stump_sgds[]=new SGD[5];
	for(int i=0;i<5;i++){
		stump_sgds[i]=new SGD(new_trains[0].numAttributes()-1);
	}
	// Train the SGDs
	for(int i=0;i<5;i++){
		stump_sgds[i].train(new_trains[i]);
	}
	// Test them on the cross validation set
	for(int i=0;i<5;i++){
		System.out.println("fold"+(i+1)+"\n");
		stump_sgds[i].test(new_tests[i]);
	}
	/***********************************************************************************************************************************************/

    }
}
