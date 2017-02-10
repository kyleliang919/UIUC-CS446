package cs446.weka.classifiers.trees;
import weka.core.Attribute;
import weka.core.Instance;
import weka.core.Instances;

import java.util.Random;
public class SGD{
	//Error threshold
	private double thresholdOfError;
	//bias
	private double w1;
	//weight
	private double[] w2;
	//learning rate
	private double r;
	//constructor
	public SGD(int num){
		w2=new double[num];
		for(int i=0;i<w2.length;i++){
			w2[i]=0;
		}
		r=0.0001;
		w1=1;
		thresholdOfError=20;
	}
	public void setLearningRate(double new_r){
		this.r=new_r;
	}
	public void setLearningBias(double new_w1){
		this.w1=new_w1;
	}
	public void setThresholdOfError(double new_thresholdOfError){
		this.thresholdOfError=new_thresholdOfError;
	}
	//tuning the learning rate and the error threshold. since I already find the optimized pair, so I exclude it from the training process
	public void parametersTuning(Instances[] trains,Instances[] tests){
		double[] rs=new double[7];
		double[] ts=new double[5];
		rs[0]=0.0001;
		rs[1]=0.0003;
		rs[2]=0.001;
		rs[3]=0.003;
		rs[4]=0.01;
		rs[5]=0.03;
		rs[6]=0.1;
		for(int i=0;i<5;i++){
			ts[i]=(i+1)*5;
		}
		// totally 35 pairs
		double[] acc=new double[35];
		double[] std=new double[35];
		for(int i=0;i<5;i++){
			for(int j=0;j<7;j++){
				double[] acc_Cv=new double[5];
				for(int k=0;k<5;k++){
					this.r=rs[j];
					this.thresholdOfError=ts[i];
					this.train(trains[k]);
					acc_Cv[k]=this.computeAccuracy(tests[k]);
					this.w2=new double[this.w2.length];
					this.w1=1;
				}
				acc[i*7+j]=mean(acc_Cv);
				std[i*7+j]=std(acc_Cv);
			}
		}
		int idx=0;
		// find the index with maximum value
		for(int i=0;i<35;i++){
			if(acc[i]>acc[idx]){
				idx=i;
			}
		}
		this.r=rs[idx%7];
		this.thresholdOfError=ts[idx/7];
		System.out.println("the best learning rate: "+rs[idx%7]);
		System.out.println("the best Error threshold: "+ts[idx/7]);
		System.out.println("its acc: "+acc[idx]);
		System.out.println("its std: "+std[idx]+"\n");

	}
	// calculate the standard deviation
	private double std(double[] samples){
		double mean=this.mean(samples);
		int num=samples.length;
		double sum=0;
		for(int i=0;i<num;i++){
			sum+=Math.pow(samples[i]-mean,2);
		}
		return Math.sqrt(sum/num);
	}
	// calculate the mean
	private double mean(double[] samples){
		int num=samples.length;
		double sum=0;
		for(int i=0;i<num;i++){
			sum+=samples[i];
		}
		return sum/num;
	}
	//train the algorithm on training set
	public void train(Instances data){
		long seed=-1;
		double currentCost=computeCost(data);
		double lastCost=0;
		while(currentCost>thresholdOfError&&Math.abs(lastCost-currentCost)>0.001){
			seed++;
			data.randomize(new Random(seed));
			for(int i=0; i<data.numInstances();i++){
				Instance sample=data.instance(i);
				double y;
				// getting the label
				if(sample.stringValue(w2.length).equals("+")){
					y=1;
				}
				else{
					y=-1;
				}
				double p=prediction(sample);
				this.updateWeight(sample,p,y);
				this.updateBias(p,y);
			}
			lastCost=currentCost;
			currentCost=computeCost(data);
		}
	}
	//prediction by the current 
	private double prediction(Instance data){
		double sum=0;
		sum+=w1;
		for(int i=0;i<w2.length;i++){
			sum+=w2[i]*data.value(i);
			//System.out.println(data.value(i));
		}
		return sum;
	}
	//compute the accuracy on given data set
	private double computeAccuracy(Instances data){
		int numOfInstances=data.numInstances();
		int numOfCorrect=0;
		int numOfFalse=0;
		for(int i=0;i<numOfInstances;i++){
			Instance sample=data.instance(i);
			double y;
			if(sample.stringValue(w2.length).equals("+")){
				y=1;
			}
			else{
				y=-1;
			}
			double p=prediction(sample);
			if((p>=0&&y>0)||(p<0&&y<0)){
				numOfCorrect++;
			}
			else{
				numOfFalse++;
			}
		}
		return (double)numOfCorrect/numOfInstances;
	}
	// compute the Error on the whole training data set
	private double computeCost(Instances data){
		double e=0;
		for(int i=0; i<data.numInstances();i++){
			Instance sample=data.instance(i);
			double y;
			if(sample.stringValue(w2.length).equals("+")){
				y=1;
			}
			else{
				y=-1;
			}
			double p=prediction(sample);
			e+=Math.pow(p-y,2);
		}
		return e/2;
	}

	//update the weight
	private void updateWeight(Instance data,double prediction,double y){
		for(int i=0;i<this.w2.length;i++){
			this.w2[i]-=r*data.value(i)*(prediction-y);
		}
	}
	//update the threshold
	private void updateBias(double prediction,double y){
		this.w1-=r*(prediction-y);
	}
	//test the algorithm on testing set
	public void test(Instances data){
		int numOfInstances=data.numInstances();
		int numOfCorrect=0;
		int numOfFalse=0;
		System.out.println("Cost on test data:"+computeCost(data));
		for(int i=0;i<numOfInstances;i++){
			Instance sample=data.instance(i);
			double y;
			if(sample.stringValue(w2.length).equals("+")){
				y=1;
			}
			else{
				y=-1;
			}
			double p=prediction(sample);
			if((p>=0&&y>0)||(p<0&&y<0)){
				numOfCorrect++;
			}
			else{
				numOfFalse++;
			}
		}
		System.out.println("number of correctly predicted instances:"+numOfCorrect);
		System.out.println("number of falsely predicted instances:"+numOfFalse);
		System.out.println("total number of instances:"+numOfInstances);
		System.out.println("accuracy:"+(double)Math.round(((double)numOfCorrect)/numOfInstances*10000)/100+"%"+"\n");		
	}
}