package cs446.homework2;

import java.io.File;
import java.util.ArrayList;
import java.util.HashSet;
import java.util.List;
import java.util.Scanner;
import java.util.Set;

import weka.core.Attribute;
import weka.core.FastVector;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.converters.ArffSaver;

public class FeatureGenerator {

    static String[] features;
    private static FastVector zeroOne;
    private static FastVector labels;

    static {
   	// ten feature types
	features = new String[] { "firstName0", "firstName1","firstName2","firstName3","firstName4","lastName0","lastName1","lastName2","lastName3","lastName4"};

	List<String> ff = new ArrayList<String>();

	for (String f : features) {
	    for (char letter = 'a'; letter <= 'z'; letter++) {
		ff.add(f + "=" + letter);
	    }
	}

	features = ff.toArray(new String[ff.size()]);

	zeroOne = new FastVector(2);
	zeroOne.addElement("1");
	zeroOne.addElement("0");

	labels = new FastVector(2);
	labels.addElement("+");
	labels.addElement("-");
    }

    public static Instances readData(String fileName) throws Exception {

	Instances instances = initializeAttributes();
	Scanner scanner = new Scanner(new File(fileName));

	while (scanner.hasNextLine()) {
	    String line = scanner.nextLine();

	    Instance instance = makeInstance(instances, line);

	    instances.add(instance);
	}

	scanner.close();

	return instances;
    }

    private static Instances initializeAttributes() {

	String nameOfDataset = "Badges";

	Instances instances;

	FastVector attributes = new FastVector(9);
	for (String featureName : features) {
	    attributes.addElement(new Attribute(featureName, zeroOne));
	}
	Attribute classLabel = new Attribute("Class", labels);
	attributes.addElement(classLabel);

	instances = new Instances(nameOfDataset, attributes, 0);

	instances.setClass(classLabel);

	return instances;

    }

    private static Instance makeInstance(Instances instances, String inputLine) {
	inputLine = inputLine.trim();

	String[] parts = inputLine.split("\\s+");
	String label = parts[0];
	String firstName = parts[1].toLowerCase();
	String lastName = parts[2].toLowerCase();

	Instance instance = new Instance(features.length + 1);
	instance.setDataset(instances);

	Set<String> feats = new HashSet<String>();
	// changes here
	if(firstName.length()<5){
		for(int i=0;i<firstName.length();i++){
			feats.add("firstName"+Integer.toString(i)+"="+firstName.charAt(i));
		}
	}
	else{
		for(int i=0;i<5;i++){
			feats.add("firstName"+Integer.toString(i)+"="+firstName.charAt(i));
		}
	}
	if(lastName.length()<5){
		for(int i=0;i<lastName.length();i++){
			feats.add("lastName"+Integer.toString(i)+"="+lastName.charAt(i));
		}
	}
	else{
		for(int i=0;i<5;i++){
			feats.add("lastName"+Integer.toString(i)+"="+lastName.charAt(i));
		}
	}
	//feats.add("firstName0=" + firstName.charAt(0));
	//feats.add("firstNameN=" + firstName.charAt(firstName.length() - 1));

	for (int featureId = 0; featureId < features.length; featureId++) {
	    Attribute att = instances.attribute(features[featureId]);

	    String name = att.name();
	    String featureLabel;
	    if (feats.contains(name)){
		featureLabel = "1";
	    } else
		featureLabel = "0";
	    instance.setValue(att, featureLabel);
	}

	instance.setClassValue(label);

	return instance;
    }

    public static void main(String[] args) throws Exception {

	if (args.length != 2) {
	    System.err
		    .println("Usage: FeatureGenerator input-badges-file features-file");
	    System.exit(-1);
	}
	Instances data = readData(args[0]);

	ArffSaver saver = new ArffSaver();
	saver.setInstances(data);
	saver.setFile(new File(args[1]));
	saver.writeBatch();
    }
}
