package edu.stanford.cs246.hw2;

import java.io.IOException;
import java.io.File;

import org.apache.hadoop.fs.Path;
import org.apache.hadoop.mapreduce.lib.input.FileInputFormat;
import org.apache.hadoop.mapreduce.lib.input.MultipleInputs;
import org.apache.hadoop.mapreduce.lib.output.FileOutputFormat;
import java.io.BufferedReader;
import java.io.DataInputStream;
import java.io.InputStreamReader;
import java.text.DecimalFormat;
import java.util.ArrayList;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.FileUtil;
import org.apache.hadoop.io.ArrayWritable;
import org.apache.hadoop.io.DoubleWritable;
import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.Job;
import org.apache.hadoop.mapreduce.Mapper;
import org.apache.hadoop.mapreduce.Reducer;
import java.net.URI;

/**
 * Algorithm:
 * 
 * Map: 
 * setup(){ 
 * 	Read the centroids 
 * } 
 * map(point) {
 * 
 * 	for each point: 
 * 		Find the nearest centroid 
 * 		emit nearest centroid :
 * 			<k,v> = <centroid id, point> 
 * 		emit cost:
 * 			 <k,v> = <-1, min_distance ^2 >
 * }
 * 
 * Reduce: 
 * setup(){ 
 * 	Read the centroids 
 * } 
 * 
 * reduce(key, points = [list of points]){ 
 * 	centroid_running_average = 0 
 * 	cost = 0 
 * 
 * 		for each point in points:
 *   		if key  == -1 :
 *   			add value to cost
 *   		else :
 * 				add point to centroid_running_average
 *  
 * 	emit cost : 
 * 		<k,v> = <C+centroid id, cost> 
 * 	emit new centroid : 
 * 		<k,v> = <K+centroid id, centroid_running_average/no of points> 
 * }
 * 
 * Main (centroid init file, input path, output dir){
 * 		for i in #jobs: 
 * 			if first iteration: 
 * 				pass centroid file path to mapper 
 * 				Run the job to get output in outputdir/i 
 * 			else: 
 * 				Merge all output in outputdir/(i-1) on HDFS to create a centroid(i) file 
 * 				pass centroid file path to mapper 
 * 				Run the job to get output in outputdir/i
 * }
 * 
 * @author sa
 * 
 */
public class KMeans {

	public static String CFILE = "cFile";
	public static String NUMOFCLUSTERS="numOfClusters";
	public static DecimalFormat df = new DecimalFormat("#.#####");

	
	
	public static class CentroidMapper extends
			Mapper<Object, Text, IntWritable, Text> {

		public ArrayList<double[]> INIT = new ArrayList<double[]>();
		@Override
		protected void setup(Context context) throws IOException,
				InterruptedException {

			//System.out.println("fs.default.name : - " + conf.get("fs.default.name"));
			// It prints uri  as : hdfs://10.214.15.165:9000 or something...
			//String uriStr = conf.get("fs.default.name");
			//String uriStr =  conf.get("fs.default.name");
			
			Configuration conf = context.getConfiguration();
			String uriStr =  "s3n://energydata/centroid/";//"./centroid";
			URI uri = URI.create(uriStr);
			FileSystem fs = FileSystem.get(uri, context.getConfiguration());		
			
			Path cFile=new Path(fs.getConf().get(CFILE));
			
			System.out.println("Mapper centroid input:" + cFile.toString());
			
			System.out.println(cFile.toString());
			DataInputStream d = new DataInputStream(fs.open(cFile));
			BufferedReader reader = new BufferedReader(new InputStreamReader(d));
			String line;
			int  numOfClusters = Integer.valueOf(context.getConfiguration().get(NUMOFCLUSTERS));
			int centroidLength=0;
			while ((line = reader.readLine()) != null) {
				if (!line.startsWith("C") && !line.startsWith("s")&&centroidLength !=numOfClusters) {
					INIT.add(parsePoint(line));
					centroidLength++;
					//System.out.println(centroidLength);
				}
			}

		}

		public void map(Object key, Text value, Context context)
				throws IOException, InterruptedException {

			if (!value.toString().startsWith("s")){ //check that its not a header
				// 1. Get point and parse it
				double[] point = parsePoint(value.toString());
				String info  = parseInfo(value.toString()); //SPID, Date;

				// 2. Get distance of point from each centroid
				int closestCentroid = 0;
				double distance = Long.MAX_VALUE;

				for (double[] centroid : INIT) {
					double tmp = distance(centroid, point);

					if (tmp < distance) {
						closestCentroid = INIT.indexOf(centroid);
						distance = tmp;
					}
				}

				// 3. Find the closest centroid for the point

				// 4. Emit cluster id and point
				context.write(new IntWritable(closestCentroid),
						new Text(longArrayToString(point)));

				//Write key=Centroid, Value=s-SPID,Date
				context.write(new IntWritable(closestCentroid),
						new Text("s-"+info));

				//Write cost value with key=-1
				context.write(new IntWritable(-1), new Text(df.format(Math.pow(distance, 2))));	
			}
		}

	}

	public static class CentroidReducer extends
			Reducer<IntWritable, Text, Text, Text> {

		public String KEY = "k";
		public String INFO = "s";
		public ArrayList<double[]> INIT = new ArrayList<double[]>();

		@Override
		protected void setup(Context context) throws IOException,
				InterruptedException {

			// Get the centroids and keep them in memory
			String uriStr =  "s3n://energydata/output/";//"./output";//;
			URI uri = URI.create(uriStr);
			FileSystem fs = FileSystem.get(uri, context.getConfiguration());	
					
			//FileSystem fs = FileSystem.get(context.getConfiguration());
			Path cFile = new Path(context.getConfiguration().get(CFILE));
			
			System.out.println("Reducer Centroid input:" + cFile.toString());

			int  numOfClusters = Integer.valueOf(context.getConfiguration().get(NUMOFCLUSTERS));
			//System.out.println(numOfClusters);
			DataInputStream d = new DataInputStream(fs.open(cFile));
			BufferedReader reader = new BufferedReader(new InputStreamReader(d));
			String line;
			int centroidLength=0;
			while ((line = reader.readLine()) != null) {
				if (!line.startsWith("C") && !line.startsWith("s")&& centroidLength !=numOfClusters) { //changed by timnit to write the document IDs for each centroid
					centroidLength++;
					//System.out.println(centroidLength);
					INIT.add(parsePoint(line));
				}
			}
			reader.close();

		}

		@Override
		protected void reduce(IntWritable key,
				Iterable<Text> values, Context context)
				throws IOException, InterruptedException {
			
			//System.out.println("Key :"+ key.get());
			//if statement added by Timnit to get SPID of people in cluster
			ArrayList<String> infoList = new ArrayList<String>();
			if (key.get() == -1) { //we are reading the cost
				double cost = 0;
				int no = 0;

				// Get average for all dimensions and cost too !
				for (Text str : values) {
						no ++;
						cost = cost + Double.parseDouble(str.toString());

				}
				//System.out.println("NO of cost data points: "+no);
				//System.out.println("Cost : "+ df.format(cost));
				context.write(new Text("C" + "-" + key.toString()), new Text(df.format(cost)));
				
			} else {

				double[] average = new double[INIT.get(0).length];
				int count = 0;
				// double cost = 0;

				// Get average for all dimensions and cost too !
				for (Text str : values) {
					if (!str.toString().startsWith("s")) {//if string has comma it is spid & date
						double[] point = parsePoint(str.toString());

						for (int i = 0; i < point.length; i++) {
							// Average for new centroid
							average[i] =  average[i]+point[i];
						}

					    count++;
					}else { //just save the spid & date
						infoList.add(str.toString());						
					}
				}

				// New centroid at center of mass for this cluster
				for (int i = 0; i < average.length; i++) {
					average[i] = average[i] / count;
				}

				// Emit new centroid
				String result = longArrayToString(average);
				String info = "";
				for (String s :infoList)
				{
					info += s+"\t";
				}				
				context.write(new Text(KEY + "-" + key.toString()), new Text(
						result));
				context.write(new Text(INFO + "-" + key.toString()), new Text(
						info));
				
			}

		}

	}

	public static void main(String[] args) throws Exception {
		
		//Run as inputDir outputDir centroidDir no_of_iters max_clusters stepSize 
		//./ output ./energy-c1_norm.txt 2 10 10

		Configuration conf = new Configuration();

		//FileSystem fs = FileSystem.get(conf);
		String inputDir = args[0];
		//String opDirBase = args[1];
		String initCentroidsPath = args[2];
		int no_of_iters = Integer.valueOf(args[3]);
		int max_clusters = Integer.valueOf(args[4]);
		int stepSize=Integer.valueOf(args[5]);
		int min_clusters = 10;
		
		
		String uriStr = inputDir;//"./input";//"s3n://energydata/input/";	
		URI uri = URI.create(uriStr);
        FileSystem fs = FileSystem.get(uri, conf);   
		System.out.println("Working directory:"+fs.getWorkingDirectory().toString());
		String inputFiles = "";
		String baseFileName    = uriStr+"_electric_interval_data_long_part";
		String suffix = "_96.txt";
		int numOfFiles = 10;
			
		for (int i=1;i<=numOfFiles;i++){
			inputFiles +=baseFileName+ String.valueOf(i) +suffix+",";
		}
		inputFiles = inputFiles.substring(0, inputFiles.length()-1);
		
		//Get all the input files in the input directory and concatenate them with a comma to input
		//into the hadoop mapper
		//Hard code the filenames for now cause Amazon EMR is having issues
		
	/*	
		String []inputFilesArray= new File(uriStr).list();
		System.out.println("Files:" + inputFilesArray.toString());
		StringBuilder fileConcatenator = new StringBuilder();
		int numOfFiles = inputFilesArray.length;
		System.out.println(numOfFiles);
	    String n = "";
		for (int i=0;i<numOfFiles;i++){
			n=inputFilesArray[i];
			fileConcatenator.append(uriStr+n).append(",");
		}
	*/	
	/*	
	 * For some reason this did not work with amazon aws and a jar file
		for (String n : inputFilesArray) {
			//fileConcatenator.append("'").append(n).append("',");
			fileConcatenator.append(uriStr+n).append(",");
		}
		
		fileConcatenator.deleteCharAt(fileConcatenator.length() - 1);
		String inputFiles = fileConcatenator.toString();
	*/	
		//System.out.println("inputs:" + inputFiles);
		
		//int j=50;
		String cDir = "";
       for (int j=min_clusters; j<=max_clusters;j+=stepSize){  //Number of clusters
        	String opDirBase = args[1]+String.valueOf(j);
        	conf.set(NUMOFCLUSTERS, String.valueOf(j));
        	//System.out.println(conf.get(NUMOFCLUSTERS));
        	System.out.println("# of Clusters:" + j
					+ "===========================================");
			for (int i = 0; i <= no_of_iters; i++) {
				System.out.println("Iteration :" + i
						+ "===========================================");
				// Output dir in HDFS for this iteration
				String outputDir = opDirBase + "/" + i;
				// System.out.println("outputDir "+i+" :"+outputDir);
				String inPath = initCentroidsPath;
	
				// Merge o/p from previous job for jobs after the init run is
				// complete
				if (i > 0) {
					cDir = opDirBase + "/" + (i - 1);
					// System.out.println("cDir "+i+" :"+cDir);
					Path inDir = new Path(cDir);
					inPath = opDirBase + "/c/" + i + "-centroid.txt";
					Path newCentroid = new Path(inPath);
		            fs.delete(newCentroid, true); //delete new centroid path if it already exists
					// Centroid file name for this iteration
					//FileUtil.copyMerge(fs, inDir, fs, new Path(inPath), false,conf, "");
		            FileUtil.copyMerge(fs, inDir, fs, newCentroid, false,conf, "");

				}
	
				// Set centroid path
				conf.set(CFILE, inPath);
				// System.out.println(conf.get(CFILE));
			
				// Job Params
				Job job = new Job(conf, "KMeans");
	
				job.setJarByClass(edu.stanford.cs246.hw2.KMeans.class);
	
				job.setMapperClass(CentroidMapper.class);
				job.setReducerClass(CentroidReducer.class);
	
				job.setMapOutputKeyClass(IntWritable.class);
				job.setOutputKeyClass(Text.class);
				job.setOutputValueClass(Text.class);
	
				//FileInputFormat.addInputPath(job, new Path(inputDir));
				//Change to have multiple inputs because input paths because energy data is patitioned into 10 txt files
				
				System.out.println("input Files:"+inputFiles);
				
				FileInputFormat.addInputPaths(job, inputFiles);	
				
				Path outputPath = new Path(outputDir);
	            fs.delete(outputPath, true); //delete output path if it already exists
				//FileOutputFormat.setOutputPath(job, new Path(outputDir));
	            FileOutputFormat.setOutputPath(job, outputPath);
			    job.waitForCompletion(true);
			    System.out.println("output Dir:" + outputPath);
			   }
        }

	}

	public static double[] parsePoint(String input) {
		double[]point = new double[96];
		int length = point.length;
		if (input.startsWith("k")) {
			String[] tk = input.split("\t");
			input = tk[1];
			String[] tokens = input.split(" ");
			//double[] point = new double[tokens.length];	
			//int length = point.length;
			for (int i = 0; i < length; i++) {
				point[i] = Double.parseDouble(tokens[i]);
			}
			
		} else{
			String[] tokens = input.split(",");
			if (tokens.length==1) {
				tokens = tokens[0].split(" ");
			}
			//double[] point = new double[96]; //[tokens.length-2-96];
			//int length = point.length;
			int offset=0;
			if (tokens[1].contains("-"))offset=2; //means we have spid & date as well
			int daily_total=0;
			for (int i=0;i<length;i++){
				point[i]=Double.parseDouble(tokens[i+offset]);
				daily_total+=point[i];
			}
			if (daily_total!=0){
				for (int i=0;i<length;i++){
					point[i]=point[i]/daily_total;
				}
			}
			
		}

		return point;
	}
	
	public static String parseInfo(String input) {
		
		//String[] tk = input.split("\t");

		String[] line = input.split(",");//not sure if tab or space is better
		String info= line[0]+","+line[1];

		return info;

	}

	// Return the Euclidean distance between 2 points in r dimensions
	public static double distance(double[] centroid, double[] point) {
		double result = 0;
		for (int i = 0; i < point.length; i++) {
			result = result + (Math.pow((point[i] - centroid[i]), 2));
			// System.out.println(result);
		}
		return Math.sqrt(result);
	}

	public static String longArrayToString(double[] average) {
		String result = new String();
		for (int i = 0; i < average.length; i++) {
			result = result + df.format(average[i]);
			if (i != average.length) {
				result = result + " ";
			}
		}
		return result;
	}

	public static class DoubleArrayWritable extends ArrayWritable {
		public DoubleArrayWritable() {
			super(DoubleWritable.class);
		}

		public DoubleArrayWritable(DoubleWritable[] values) {
			super(DoubleWritable.class, values);
		}
	}

}
