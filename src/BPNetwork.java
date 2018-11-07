import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.io.BufferedWriter;
import java.io.FileWriter;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.List;
import java.util.Random;
import java.util.StringTokenizer;
import java.lang.Math;

public class BPNetwork {

	static int numData = 2513;
	static int numTrain = (int) (numData*0.7);
	static int numVal = (int) (numData*0.15);
	static int numTest = numVal;
	static int numInputs = 11;
	static int numOutputs = 3;
	static int numHidden = 10;
	static double lRate = 0.2;
	static double alpha = 0;
	static double threshold = 0.05;
	
	static double[][] trainInputs = new double[numTrain][numInputs+1];
	static double[][] testInputs = new double[numTest][numInputs +1];
	static double[][] trainOutputs = new double[numTrain][numOutputs];
	static double[][] testOutputs = new double[numTest][numOutputs];
	static double[][] valInputs = new double [numVal][numInputs+1];
	static double[][] valOutputs = new double[numVal][numOutputs];
	
	static double[] hiddenOutput = new double[numHidden];
	static double[] hiddenTest = new double[numTest];
	static double[][] hiddenWeights = new double[numHidden][numInputs + 1];
	static double[][] initialHiddenWeights = new double[numHidden][numInputs + 1];
	
	static double[][] resultOutputs = new double[503][3];
	static double[] outputs = new double [numOutputs];
	static double[] saveTestOutputs = new double[numTest];
	static double[][] outputWeights = new double[numOutputs][numHidden + 1];
	static double[][] initialOutputWeights = new double[numOutputs][numHidden + 1];
	
	static double[] outputError = new double[numOutputs];
	static double[] dhiddenError = new double[numHidden];
	
	static double trainError;
	static double valError;
	static double[][] finalOutputs = new double[numTest][numInputs+2];
	
	public static void main(String args []) throws IOException {
		
		//import training data
			importData("assignment2data.csv");
			
			initializeWeights();
			feedForward();
			
			
			
			for (int i = 0; i<numOutputs; i++) {
				for (int j =0; j<numHidden +1; j++) {
					//System.out.print(outputWeights[i][j] + ", ");
				}
				//System.out.println(" ");
			}
			testResults();
			finalOutput();
			writeTextFile("Final_Result.txt", finalOutputs);
	}
	
	private static void testResults() {
		for (int x=0; x<numTest; x++) {
			valError = 0;
			for (int i=0; i<numHidden; i++) {
				for (int j=0; j<numInputs+1; j++) {
					//System.out.println(valInputs[x][j]);
					hiddenOutput[i] += hiddenWeights[i][j]*valInputs[x][j];
				}
				hiddenOutput[i] = sigFcn(hiddenOutput[i]);
				//System.out.println(hiddenOutput[i]);
				}
		
			//calculate output node outputs
			for (int i=0; i<numOutputs; i++) {
				for (int j=0; j<numHidden; j++) {
					outputs[i] += hiddenOutput[j]*outputWeights[i][j];
				}
				outputs[i] += outputWeights[i][numHidden];
				outputs[i] = sigFcn(outputs[i]);
				if (outputs[i] > 1-threshold)
					outputs[i] = 1;
				else if (outputs[i] < threshold)
					outputs[i] = 0;
					}
			
			if (resultOutputs[x][0] == 1 && resultOutputs[x][1] == 0 && resultOutputs[x][2] == 0)
				finalOutputs[x][numInputs +1] = 5;
			else if(resultOutputs[x][0] == 0 && resultOutputs[x][1] == 1 && resultOutputs[x][2] == 0)
				finalOutputs[x][numInputs +1] = 7;
			else if(resultOutputs[x][0] == 0 && resultOutputs[x][1] == 0 && resultOutputs[x][2] == 1)
				finalOutputs[x][numInputs+1] = 8;
			
		}		
		
	}

	private static void finalOutput() {
		for (int i=0; i<numTest; i++) {
			for (int j=0; j<numInputs; j++) {
				finalOutputs[i][j] = testInputs[i][j];
			}
			finalOutputs[i][numInputs] = saveTestOutputs[i];
			
			for(int x=0; x<numInputs+2; x++) {
					System.out.print(finalOutputs[i][x] + ", ");
				}
			System.out.println("");
		}
		}
		
		

	private static void feedForward() {
		int y = 0;
		valError = 5;
		double valErrorTemp =100;
		do {
			for (int x=0; x<numTrain; x++) {
				for (int i=0; i<numHidden; i++) {
					for (int j=0; j<numInputs+1; j++) {
						hiddenOutput[i] += hiddenWeights[i][j]*trainInputs[x][j];
					}
					hiddenOutput[i] = sigFcn(hiddenOutput[i]);
				}
			
				//calculate output node outputs
				for (int i=0; i<numOutputs; i++) {
					for (int j=0; j<numHidden; j++) {
						outputs[i] += hiddenOutput[j]*outputWeights[i][j];
					}
					outputs[i] += outputWeights[i][numHidden];
					outputs[i] = sigFcn(outputs[i]);
					/*if (outputs[i] > 1-threshold)
						outputs[i] = 1;
					else if (outputs[i] < threshold)
						outputs[i] = 0;*/
				}
			
				backpropogate(x);
			}
			valErrorTemp = valError + 0.0001;
			validate();
			y++;
			//System.out.println(trainError + ", " + y + ", " + valError + ", " + valErrorTemp);
		} while (valError < valErrorTemp || y < 1500);
			
	}


	private static void validate() {
		
		for (int x=0; x<numVal; x++) {
			valError = 0;
			for (int i=0; i<numHidden; i++) {
				for (int j=0; j<numInputs+1; j++) {
					//System.out.println(valInputs[x][j]);
					hiddenOutput[i] += hiddenWeights[i][j]*valInputs[x][j];
				}
				hiddenOutput[i] = sigFcn(hiddenOutput[i]);
				//System.out.println(hiddenOutput[i]);
				}
		
			//calculate output node outputs
			for (int i=0; i<numOutputs; i++) {
				for (int j=0; j<numHidden; j++) {
					outputs[i] += hiddenOutput[j]*outputWeights[i][j];
				}
				outputs[i] += outputWeights[i][numHidden];
				outputs[i] = sigFcn(outputs[i]);
				if (outputs[i] > 1-threshold)
					outputs[i] = 1;
				else if (outputs[i] < threshold)
					outputs[i] = 0;
			}
			for(int i=0; i<numOutputs; i++) {
				outputError[i] = valOutputs[x][i] - outputs[i];
				valError += outputError[i]*outputError[i]/2;
				//System.out.println(valError);
			}
		}
	}

	private static void backpropogate(int index) {
		trainError = 0;
		//calculate output error
		for(int i=0; i<numOutputs; i++) {
			//System.out.println(outputs[i]);
			outputError[i] = trainOutputs[index][i] - outputs[i];
			trainError += outputError[i]*outputError[i]/2;
		}
		
		//calculate dj error for the hidden layer
		for (int i=0; i<numHidden; i++) {
			dhiddenError[i] = 0;
			for (int j=0; j<numOutputs; j++) {
				dhiddenError[i] += outputError[j]*outputs[j]*(1-outputs[j])*outputWeights[j][i];
			}
		}
		
		//update output node weights
		for (int i=0; i<numOutputs; i++) {
			for (int j=0; j<numHidden; j++) {
				outputWeights[i][j] += lRate*hiddenOutput[j]*outputError[i]*outputs[i]*(1-outputs[i]) + alpha*outputWeights[i][j];
				//System.out.println(outputWeights[i][j]);
			}
			outputWeights[i][numHidden] += lRate*outputError[i]*outputs[i]*(1-outputs[i]) + alpha*outputWeights[i][numHidden];
		
			}
		
		//update hidden node weights
		for (int i=0; i<numHidden; i++) {
			for (int j=0; j<numInputs+1; j++) {
				hiddenWeights[i][j] += lRate*trainInputs[index][j]*dhiddenError[i]*hiddenOutput[i]*(1-hiddenOutput[i]) + alpha*hiddenWeights[i][j];
			}
		}
	}

	private static double sigFcn(double f) {
		
		return 1/(1 + Math.exp(-f));
	}

	private static void initializeWeights() {
		Random rand = new Random();
		for(int i=0; i<numHidden; i++) {
			for (int j=0; j<numInputs+1; j++) {
				hiddenWeights[i][j] = (double) ((rand.nextInt(100)+1)/100.00);
				initialHiddenWeights[i][j] = hiddenWeights[i][j];
			}
		}
		
		for(int i=0; i<numOutputs; i++) {
			for (int j=0; j<numHidden +1; j++) {
				outputWeights[i][j] = (double) ((rand.nextInt(100)+1)/100.00);
				initialOutputWeights[i][j] = outputWeights[i][j];
			}
		}
		
	}

	public static void importData(String fileName) throws IOException {
		
		List<String> Data = Files.readAllLines(Paths.get(fileName));
		stringToArray(Data);
		trainInputs = preprocessInputs(trainInputs);
		testInputs = preprocessInputs(testInputs);
		valInputs = preprocessInputs(valInputs);
	}
		
	//Reads the input file list and converts the data into their respective arrays
	static void stringToArray(List<String> list){

		String[] stringArray = list.toArray(new String[0]);
		double temp; 
		
		for (int i=0; i< numTrain; i++) {
			
			StringTokenizer tokenizer = new StringTokenizer(stringArray[i+1], ",");
			
			for (int j=0; j<numInputs; j++) {
				trainInputs[i][j] = Double.parseDouble(tokenizer.nextToken());
			}
			temp = Double.parseDouble(tokenizer.nextToken());
			
			if (temp == 5) {
				trainOutputs[i][0] = 1;
				trainOutputs[i][1] = 0;
				trainOutputs[i][2] = 0;
			}
			else if (temp == 7) {
				trainOutputs[i][0] = 0;
				trainOutputs[i][1] = 1;
				trainOutputs[i][2] = 0;
			}
			else if (temp == 8) {
				trainOutputs[i][0] = 0;
				trainOutputs[i][1] = 0;
				trainOutputs[i][2] = 1;	
			}
		}
		
		for (int i=0; i< numTest; i++) {
			
			StringTokenizer tokenizer = new StringTokenizer(stringArray[i+numTrain], ",");
			
			for (int j=0; j<numInputs; j++) {
			testInputs[i][j] = Double.parseDouble(tokenizer.nextToken());
			}
			temp = Double.parseDouble(tokenizer.nextToken());
			saveTestOutputs[i] = temp;
			
			if (temp == 5) {
				testOutputs[i][0] = 1;
				testOutputs[i][1] = 0;
				testOutputs[i][2] = 0;
			}
			else if (temp == 7) {
				testOutputs[i][0] = 0;
				testOutputs[i][1] = 1;
				testOutputs[i][2] = 0;
			}
			else if (temp == 8) {
				testOutputs[i][0] = 0;
				testOutputs[i][1] = 0;
				testOutputs[i][2] = 1;
			}
		}
		
	for (int i=0; i< numVal; i++) {
			
			StringTokenizer tokenizer = new StringTokenizer(stringArray[i+numTrain+numTest], ",");
			
			for (int j=0; j<numInputs; j++) {
			valInputs[i][j] = Double.parseDouble(tokenizer.nextToken());
			}
			temp = Double.parseDouble(tokenizer.nextToken());
			
			if (temp == 5) {
				valOutputs[i][0] = 1;
				valOutputs[i][1] = 0;
				valOutputs[i][2] = 0;
			}
			else if (temp == 7) {
				valOutputs[i][0] = 1;
				valOutputs[i][1] = 1;
				valOutputs[i][2] = 0;
			}
			else if (temp == 8) {
				valOutputs[i][0] = 1;
				valOutputs[i][1] = 1;
				valOutputs[i][2] = 1;
			}
	}
	}
	
	static double[][] preprocessInputs(double[][] list) {
		double[] max = new double[11];
		double[] min = new double[11];
		List<Double> temp = new ArrayList<Double>();
		
		//find max and min values for each attribute
		for (int i = 0; i<11; i++) {
			for (int j = 0; j<list.length; j++) {
				temp.add(list[j][i]);
			}
			max[i] = Collections.max(temp);
			min[i] = Collections.min(temp);
		}
		
		//normalize each input
		for (int i = 0; i<list.length; i++) {
			for (int j = 0; j<11; j++) {
				//System.out.println(list.get(i)[j]);
				list[i][j] = (list[i][j] - min[j])/(max[j] - min[j]);
			}
			list[i][list[i].length-1] = 1;
		}
		return list;
	}
	
	static void writeTextFile(String filename, double[][] list) {
		try {
			BufferedWriter bw = new BufferedWriter(new FileWriter(filename));
			
			bw.write("There were 10 hidden nodes and 3 output nodes. These are the initial hidden node weights:");
			bw.newLine();
			for (int i=0; i<numHidden; i++) {
				for (int j=0; j<numInputs+1; j++) {
					bw.write(initialHiddenWeights[i][j] + ((j == numInputs) ? "" : ","));
				}
				bw.newLine();
			}
			bw.write("These are the initial output node weights:");
			bw.newLine();
			for (int i=0; i<numOutputs; i++) {
				for (int j=0; j<numHidden+1; j++) {
					bw.write(initialOutputWeights[i][j] + ((j == numHidden) ? "" : ","));
				}
				bw.newLine();
			}
			bw.write("Training Root Mean Squared Error: " + trainError);
			bw.newLine();
			bw.write("Validation Root Mean Squared Error: " + valError);
			bw.newLine();
			bw.write("Final weight vectors for the output nodes:");
			bw.newLine();
			for (int i=0; i< numOutputs; i++) {
				for (int j=0; j<numHidden+1; j++) {
					bw.write(outputWeights[i][j] + (j == numHidden ? "" : ","));
				}
				bw.newLine();
			}
			bw.newLine();
			bw.write("Final weight vectors for the hidden nodes:");
			bw.newLine();
			for (int i=0; i< numHidden; i++) {
				for (int j=0; j<numInputs+1; j++) {
					bw.write(hiddenWeights[i][j] + (j == numInputs ? "" : ","));
				}
				bw.newLine();
			}
			bw.newLine();
			for (int i = 0; i<list.length; i++) {
				for (int j = 0; j<list[i].length; j++) {
					bw.write(list[i][j] + ((j == list[i].length-1) ? "" : ","));
				}
				bw.newLine();
			}
			bw.flush();		
		}catch (IOException e) {}
	}
}
