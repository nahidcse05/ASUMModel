/*
 * Implementation of Sentence Topic/Opinion
 *   - Different THETAs for different sentiments: THETA[S]
 *   - Positive/Negative
 * Author: Yohan Jo
 */
package sto2;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileInputStream;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.IOException;
import java.io.InputStreamReader;
import java.io.PrintWriter;
import java.lang.reflect.Array;
import java.util.ArrayList;
import java.util.Date;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Random;
import java.util.Set;
import java.util.TreeSet;
import java.util.Vector;

import structures.MyPriorityQueue;
import structures._RankItem;
import util.DoubleMatrix;
import util.IntegerMatrix;
import util.OrderedDocument;
import util.Sentence;
import util.SentiWord;
import util.Utility;
import util.Word;

public class STO2Core1 {
	private int numUniqueWords;
	private int numTopics;
	private int numSenti;
	private int numRealIterations;
	private int numDocuments;
	private List<String> wordList = null;
	private int numProbWords = 200;
	private boolean sentiAspectPrior = false;
	
	public String inputDir = null;
	public String outputDir = null;
	private Integer intvalTmpOutput = null;
	
	int mod;
	private double alpha;
	private double[][] word_topic_prior; /* prior distribution of words under a set of topics, by default it is null */
	
	private double sumAlpha;
	private double [] betas;  // betas[3]: Common Words, Corresponding Lexicon, The Other Lexicons
	private double [] sumBeta;  // sumBeta[senti]
	private double [] gammas;
	private double sumGamma;
	
	public DoubleMatrix [] Phi; // Phi[senti][word][topic]
	public DoubleMatrix [] Theta;  // Theta[senti][document][topic]
	public DoubleMatrix Pi;
	
	public List<TreeSet<Integer>> sentiWordsList;
	
	private IntegerMatrix [] matrixSWT;
	private IntegerMatrix [] matrixSDT;
	private IntegerMatrix matrixDS;
	
	private int[][] sumSTW;  // sumSTW[S][T]
	private int[][] sumDST;  // sumDST[D][S]
	private int[] sumDS;  // sumDS[D]
	
	private double[][] probTable;
	
	private List<OrderedDocument> documents;
	final private int maxSentenceLength = 50;
	
	public  PrintWriter infoWriter;
	public  PrintWriter summaryWriter;
	private  String wordIntrusionFilePath;
	
	public void setsentiAspectPrior(boolean flag){
		sentiAspectPrior = flag;
	}
	
	public void setInfoWriter(String filePath){
		try{
			infoWriter = new PrintWriter(new File(filePath));
			
			System.out.println("File Set");
		}
		catch(Exception e){
			e.printStackTrace();
			System.err.println("Info file"+filePath+" Not Found");
		}
	}
	
	public void setSummaryWriter(String filePath){
		try{
			
			summaryWriter = new PrintWriter(new File(filePath));
			System.out.println("File Set");
		}
		catch(Exception e){
			e.printStackTrace();
			System.err.println("Summary file"+filePath+" Not Found");
		}
	}
	
	
	public void setIntrusionWriter(String filePath){
		this.wordIntrusionFilePath = filePath;
	}
	
	
	public void LoadPrior(String filename, int eta) {		
		if (filename == null || filename.isEmpty())
			return;
		
		try {
			String tmpTxt;
			String[] container;
			
			HashMap<String, Integer> featureNameIndex = new HashMap<String, Integer>();
			for(int i=0; i<wordList.size(); i++)
				featureNameIndex.put(wordList.get(i), featureNameIndex.size());
			
			int wid, wCount = 0;
			
			double[] prior;
			ArrayList<double[]> priorWords = new ArrayList<double[]>();
			BufferedReader reader = new BufferedReader(new InputStreamReader(new FileInputStream(filename), "UTF-8"));
			while( (tmpTxt=reader.readLine()) != null ){
				tmpTxt = tmpTxt.trim();
				if (tmpTxt.isEmpty())
					continue;
				
				System.out.println("Prior loaded:" + tmpTxt);
				int uniGramCount = 0; 
				int biGramCount = 0;
				container = tmpTxt.split(" ");
				wCount = 0;
				prior = new double[wordList.size()];
				for(int i=1; i<container.length; i++) {
					//here for checking we have added the stemming
					// but stemming is reducing the number of prior loaded
					// so we turn off the loading of stemming
					//container[i] = SnowballStemming(container[i]); // stemmer added
					if (featureNameIndex.containsKey(container[i])) {
						wid = featureNameIndex.get(container[i]); // map it to a controlled vocabulary term
						prior[wid] = eta;
						wCount++;
						if(container[i].contains("-"))
							biGramCount++;
						else
							uniGramCount++;
					}
					
				}
				System.out.format("Prior keywords for Topic %d (%s): %d/%d\n", priorWords.size(), container[0], wCount, container.length-1);
				System.out.println("Unigram loaded:"+uniGramCount +", Bigram loaded:"+biGramCount);
				priorWords.add(prior);
			}
			reader.close();
			
			word_topic_prior = priorWords.toArray(new double[priorWords.size()][]);
			
			/*if (m_sentiAspectPrior && word_topic_prior.length%2==1) {
				System.err.format("The topic size (%d) specified in the sentiment-aspect seed words is not even!", word_topic_prior.length);
				System.exit(-1);
			} else if (word_topic_prior.length > number_of_topics) {
				System.err.format("More topics specified in seed words (%d) than topic model's configuration(%d)!\n", word_topic_prior.length, number_of_topics);
				System.err.format("Reset the topic size to %d!\n", word_topic_prior.length);
				
				this.number_of_topics = word_topic_prior.length;
				createSpace();
			}
			*/
		} catch (IOException e) {
			e.printStackTrace();
		}
	}

	
	protected void imposePrior() {		
		if (word_topic_prior!=null) {//we have enforced that the topic size is at least as many as prior seed words
			
			int s = 0;
			for(int k=0; k<numTopics; k++) {
				for(int n=0; n<wordList.size(); n++) {
					matrixSWT[s].setValue(n, k, (int)word_topic_prior[k][n]);
				}
			}
			
			s = 1;
			for(int k=numTopics; k<2*numTopics; k++) {
				for(int n=0; n<wordList.size(); n++) {
					matrixSWT[s].setValue(n, k-numTopics, (int)word_topic_prior[k][n]);
				}
			}

		}
	}
	
	
	public static void main(String [] args) throws Exception {
		
		int numIterations = 500;
		int numSenti = 2;
		int numThreads = 1;
		
		//String[] products = {"camera","tablet", "laptop", "phone", "surveillance", "tv"};
		String category = "tablet";
		int numTopics = 15;  // here the number of topics half of LRHTSM
		int trainSize = 5000;
		boolean neweggload = true;
		boolean loadSentiPrior = true;
		boolean loadSentiAspectPrior = true;
		
		String inputDir = null;
		String outputDir = null;
		
		
		if(neweggload){
			inputDir = "./data/input/"+ trainSize+"/" +category+"/";
			outputDir = "./data/output/"+ trainSize+"/"+category+"/";
		}
		else{
			inputDir = "./data/input/nonewegg/"+ trainSize+"/" +category+"/";
			outputDir = "./data/output/nonewegg/"+ trainSize+"/"+category+"/";
		}
		String dicDir = "./data/input/";
		String resultPath = outputDir+category+"_information.txt";
		String summaryPath = outputDir+"ASUM_" +category+"_Topics_" + numTopics*2 + "_Summary.txt";
		String wordIntrusionFilePath = outputDir+ "ASUM" +"_" +category+"_Topics_" + numTopics*2 + "_WordIntrusion.txt";
		
		String aspectSentiList = "./data/Model/aspect_"+ category + ".txt";
		
		
		double alpha = 0.01;
		double [] betas = new double[3];
		int eta = 5;
		betas[0] = 0.001;
		betas[1] = 0.1;
		
		double [] gammas =  new double[numSenti];
		gammas[0] = 1; gammas[1] = 1;
		String [] betasStr = null;
		String [] gammasStr = null;
		boolean randomInit = false;
		
		String sentiFilePrefix = null;
		
		if(loadSentiPrior){
			sentiFilePrefix = "SentiWords-";
		}else{
			sentiFilePrefix = "NoSentiWords-";
		}
		
		String wordListFileName = "selected_combine_fv.txt";
		String wordDocFileName = "BagOfSentences_pros_cons.txt";

		if (inputDir == null) inputDir = ".";
		if (outputDir == null) outputDir = new String(inputDir);
		if (dicDir == null) dicDir = new String(inputDir);

		// Exceptions
		if (!new File(inputDir).exists()) throw new Exception("There's no such an input directory as " + inputDir);
		if (!new File(outputDir).exists()) throw new Exception("There's no such an output directory as " + outputDir);
		if (!new File(dicDir).exists()) throw new Exception("Tehre's no such a dictionary directory as " + dicDir);
		
		
		String line;		
		Vector<String> wordList = new Vector<String>();
		BufferedReader wordListFile = new BufferedReader(new FileReader(new File(inputDir+"/"+wordListFileName)));
		while ((line = wordListFile.readLine()) != null)
			if (line != "") wordList.add(line);
		wordListFile.close();
		
		Vector<OrderedDocument> documents = OrderedDocument.instantiateOrderedDocuments(inputDir+"/"+wordDocFileName, null, null);

		ArrayList<TreeSet<String>> sentiWordsStrList = new ArrayList<TreeSet<String>>();
		for (int s = 0; s < numSenti; s++) {
			String dicFilePath = dicDir + "/" + sentiFilePrefix+s+".txt"; 
			if (new File(dicFilePath).exists()) {
				sentiWordsStrList.add(Utility.makeSetOfWordsFromFile(dicFilePath, true));
			}
		}
		
		ArrayList<TreeSet<Integer>> sentiWordsList = new ArrayList<TreeSet<Integer>>(sentiWordsStrList.size());
		for (Set<String> sentiWordsStr : sentiWordsStrList) {
			TreeSet<Integer> sentiWords = new TreeSet<Integer>();
			for (String word : sentiWordsStr)
				sentiWords.add(wordList.indexOf(word));
			sentiWordsList.add(sentiWords);
		}
		
		// Print the configuration
		System.out.println("Documents: "+documents.size());
		System.out.println("Unique Words: "+wordList.size());
		System.out.println("Topics: "+numTopics);
		System.out.println("Sentiments: "+numSenti+" (dictionary: "+sentiWordsList.size()+")");
		System.out.println("Alpha: "+alpha);

		System.out.println("Iterations: "+numIterations);
		System.out.println("Threads: "+numThreads);
		System.out.println("Input Dir: "+inputDir);
		System.out.println("Dictionary Dir: "+dicDir);
		System.out.println("Output Dir: "+outputDir);
		
		STO2Core1 core = new STO2Core1(numTopics, numSenti, wordList, documents, sentiWordsList, alpha, betas, gammas);
		if(loadSentiAspectPrior){
			System.out.println("Loading aspect-senti list from "+aspectSentiList);
			core.setsentiAspectPrior(loadSentiAspectPrior);
			core.LoadPrior(aspectSentiList, eta);
		} 
		core.setInfoWriter(resultPath);
		core.setSummaryWriter(summaryPath);
		core.setIntrusionWriter(wordIntrusionFilePath);
		core.generateTmpOutputFiles(inputDir, outputDir, 1000);
		core.initialization(randomInit);
		core.gibbsSampling(numIterations, numThreads);
		core.generateOutputFiles(outputDir);
		core.sampleTestdoc();
		
		// most popular items under each category from Amazon
		String tabletProductList[] = {"B008DWG5HE","B00CYQPM42","B007P4YAPK"};
		String cameraProductList[] = {"B005IHAIMA","B002IPHIEG","B00DMS0LCO"};
		String phoneProductList[] = {"B00COYOAYW","B004T36GCU","B008HTJLF6"};
		String tvProductList[] = {"B0074FGLUM","B00BCGROJG","B00AOA9BL0"};
		
		if(category.equalsIgnoreCase("camera"))
			core.docSummary(cameraProductList);
		else if(category.equalsIgnoreCase("tablet"))
			core.docSummary(tabletProductList);
		else if(category.equalsIgnoreCase("phone"))
			core.docSummary(phoneProductList);
		else if(category.equalsIgnoreCase("tv"))
			core.docSummary(tvProductList);
	}
	
	public STO2Core1(int numTopics, int numSenti, List<String> wordList, List<OrderedDocument> documents, List<TreeSet<Integer>> sentiWordsList, double alpha, double[] betas, double [] gammas) {
		this.numTopics = numTopics;
		this.numSenti = numSenti;
		this.numUniqueWords = wordList.size();
		this.numDocuments = documents.size();
		this.documents = documents;
		this.wordList = wordList;
		this.sentiWordsList = sentiWordsList;
		this.alpha = alpha;
		this.betas = betas;
		this.gammas = gammas;
		this.sumBeta = new double[numSenti];
		probTable = new double[numTopics][numSenti];
	}
	
	

	public void initialization(boolean randomInit) {
		sumSTW = new int[numSenti][numTopics];
		sumDST = new int[numDocuments][numSenti];
		sumDS = new int[numDocuments];
		
		matrixSWT = new IntegerMatrix[numSenti];
		for (int i = 0; i < numSenti; i++)
			matrixSWT[i] = new IntegerMatrix(numUniqueWords, numTopics);
		matrixSDT = new IntegerMatrix[numSenti];
		for (int i = 0; i < numSenti; i++)
			matrixSDT[i] = new IntegerMatrix(numDocuments, numTopics);
		matrixDS = new IntegerMatrix(numDocuments, numSenti);
		
		if(this.sentiAspectPrior==true){
			System.out.println("Imposing Prior");
			imposePrior();
		}
		
		
		int numTooLongSentences = 0;

		for (OrderedDocument currentDoc : documents){
			int docNo = currentDoc.getDocNo();
			
			for (Sentence sentence : currentDoc.getSentences()) {
				int newSenti = -1;
				int numSentenceSenti = 0;
				for (Word sWord : sentence.getWords()) {
					SentiWord word = (SentiWord) sWord;
					
					int wordNo = word.getWordNo();
					for (int s = 0; s < sentiWordsList.size(); s++) {
						if (sentiWordsList.get(s).contains(wordNo)) {
							if (numSentenceSenti == 0 || s != newSenti) numSentenceSenti++;
							word.lexicon = s;
							newSenti = s;
						}
					}
				}
				sentence.numSenti = numSentenceSenti;
				
				if (randomInit || sentence.numSenti != 1)
					newSenti = (int)(Math.random()*numSenti);
				int newTopic = (int)(Math.random()*numTopics);

				if (sentence.getWords().size() > this.maxSentenceLength) numTooLongSentences++;
				
				if (!(numSentenceSenti > 1 || sentence.getWords().size() > this.maxSentenceLength)) {
					sentence.setTopic(newTopic);
					sentence.setSenti(newSenti);
					
					for (Word sWord : sentence.getWords()) {
						((SentiWord) sWord).setSentiment(newSenti);
						sWord.setTopic(newTopic);
						matrixSWT[newSenti].incValue(sWord.wordNo, newTopic);
						sumSTW[newSenti][newTopic]++;
					}
					matrixSDT[newSenti].incValue(docNo, newTopic);
					matrixDS.incValue(docNo, newSenti);
	
					sumDST[docNo][newSenti]++;
					sumDS[docNo]++;
				}
			}
		}
		
		System.out.println("Too Long Sentences: "+numTooLongSentences);
	}
	
	public void gibbsSampling(int numIterations, int numThreads) throws Exception {
		this.sumAlpha = this.alpha * this.numTopics;
		int numSentiWords = 0;
		for (Set<Integer> sentiWords : sentiWordsList) numSentiWords += sentiWords.size();
		double sumBetaCommon = this.betas[0] * (this.numUniqueWords - numSentiWords);
		for (int s = 0; s < numSenti; s++) {
			int numLexiconWords = 0;
			if (this.sentiWordsList.size() > s) numLexiconWords = this.sentiWordsList.get(s).size();
			this.sumBeta[s] = sumBetaCommon + this.betas[1]*numLexiconWords + this.betas[2]*(numSentiWords-numLexiconWords);
		}
		this.sumGamma = 0;
		for (double gamma : this.gammas) this.sumGamma += gamma;
		
		System.out.println("Gibbs sampling started (Iterations: "+numIterations+", Threads: "+numThreads+")");
		
		long startTime, endTime;
		for(int i = 0; i < numIterations; i++){
			
			System.out.println( "  - Iteration " + i);

			for (Set<Integer> sentiWords : this.sentiWordsList) {
				for (int wordNo : sentiWords) {
					if (wordNo < 0 || wordNo >= this.wordList.size()) continue;
					System.out.print(this.wordList.get(wordNo)+"/");
					for (int s = 0; s < numSenti; s++) {
						int sum = 0;
						for (int t = 0; t < numTopics; t++) sum += matrixSWT[s].getValue(wordNo, t);
						System.out.print(sum+"/");
					}
					System.out.print(" ");
				}
				System.out.println();
			}
			
			startTime = new Date().getTime();
			for (OrderedDocument currentDoc : documents){
				if(!currentDoc.isTestDoc())
					sampleForDoc(currentDoc);
			}
			endTime = new Date().getTime();
			double seconds = (int)(endTime - startTime)/1000.0;
			int minutes = (int)(seconds * (numIterations - i - 1) / 60);
			System.out.println("    Iteration "+i+" took "+seconds+"s. (Estimated Time: "+(minutes/60)+"h "+(minutes%60)+"m)");
			
			this.numRealIterations = i + 1;
			if (this.intvalTmpOutput != null && this.numRealIterations % this.intvalTmpOutput == 0 && this.numRealIterations < numIterations) {
				this.Phi = STO2Util.calculatePhi(matrixSWT, sumSTW, this.betas, this.sumBeta, this.sentiWordsList);
				this.Theta = STO2Util.calculateTheta(matrixSDT, sumDST, this.alpha, this.sumAlpha);
				this.Pi = STO2Util.calculatePi(matrixDS, sumDS, this.gammas, this.sumGamma);
				generateOutputFiles(this.outputDir);
			}
		}
		System.out.println("Gibbs sampling terminated.");
		
		this.Phi = STO2Util.calculatePhi(matrixSWT, sumSTW, this.betas, this.sumBeta, this.sentiWordsList);
		this.Theta = STO2Util.calculateTheta(matrixSDT, sumDST, this.alpha, this.sumAlpha);
		this.Pi = STO2Util.calculatePi(matrixDS, sumDS, this.gammas, this.sumGamma);
	}

	public void sampleTestdoc() throws IOException
	{
		
		String dir = "./data/output/";
		PrintWriter out = new PrintWriter(new FileWriter(new File(dir +"VisReviews.html")));
		double perplexity = 0.0;
		double likelihood = 0.0;
		int number_of_test_doc = 0;
		int numIterations = 100; // Gibbs iteration
		double log2 = Math.log(2.0);
		int precision_recall [][] = new int[2][2]; // row for true label and col for predicted label 
		int precision_recall_sentences [][] = new int[2][2]; // row for true label and col for predicted label 
		// row for real label, col is for predicted label
		precision_recall[0][0] = 0; // 0 is for pos
		precision_recall[0][1] = 0; // 1 is neg 
		precision_recall[1][0] = 0;
		precision_recall[1][1] = 0;
		precision_recall_sentences[0][0] = 0; // 0 is for pos
		precision_recall_sentences[0][1] = 0; // 1 is neg
		precision_recall_sentences[1][0] = 0;
		precision_recall_sentences[1][1] = 0;
		int cannotPredict = 0;
		
		for(int i=0; i<numIterations;i++){
			for (OrderedDocument doc : documents){
				if(doc.isTestDoc()){ // true means this is a test doc
					sampleForDoc(doc);
				}
			}
			this.Theta = STO2Util.calculateTheta(matrixSDT, sumDST, this.alpha, this.sumAlpha);
			this.Pi = STO2Util.calculatePi(matrixDS, sumDS, this.gammas, this.sumGamma);
			//this.Phi = STO2Util.calculatePhi(matrixSWT, sumSTW, this.betas, this.sumBeta, this.sentiWordsList);
		}
		
		
		for (OrderedDocument doc : documents)
		{
			if(doc.isTestDoc())
			{
				number_of_test_doc++;
	
				String [] sentiColors = {"green","red","black"};
				
				out.println("<h3>Real Document "+doc.getDocNo()+"</h3>");
				for (Sentence sentence : doc.getSentences()) {
					if (sentence.getSenti() < 0 || sentence.getSenti() >= this.numSenti || sentence.getWords().size() > this.maxSentenceLength) continue;
					out.print("<p style=\"color:"+sentiColors[sentence.label]+";\">T"+sentence.getTopic()+":");
					for (Word word : sentence.getWords()) out.print(" "+this.wordList.get(word.wordNo));
					out.println("</p>");
				}
				
				
				out.println("<h3>Predicted Document "+doc.getDocNo()+"</h3>");
				for (Sentence sentence : doc.getSentences()) {
					if (sentence.getSenti() < 0 || sentence.getSenti() >= this.numSenti || sentence.getWords().size() > this.maxSentenceLength) 
					{
						cannotPredict++;
						continue;
					}
					out.print("<p style=\"color:"+sentiColors[sentence.getSenti()]+";\">T"+sentence.getTopic()+":");
					if(sentence.label<=1){
						precision_recall_sentences[sentence.label][sentence.getSenti()]++;

						for (Word word : sentence.getWords())
						{
							out.print(" "+this.wordList.get(word.wordNo));
							precision_recall[sentence.label][sentence.getSenti()]++;

						}
						out.println("</p>");
					}
				}
				
			
			// perplexity calculation
//			int d = doc.getDocNo();
//			double val = 0.0;
//			int doc_lenght = 0;
//			for (Sentence sentence : doc.getSentences()) {
//				
//				for (Word word : sentence.getWords()){
//					double prob = 0.0; 
//					int w = word.wordNo;
//					doc_lenght++;
//					for (int s = 0; s < numSenti; s++) {
//						for (int t = 0; t < numTopics; t++) {
//							prob += this.Phi[s].getValue(w, t)*((matrixSDT[s].getValue(d,t) + alpha) / (sumDST[d][s] + sumAlpha))*((matrixDS.getValue(d,s) + gammas[s]) / (sumDS[d] + sumGamma));
//							//prob += (matrixSWT[s].getValue(w, t)) / (sumSTW[s][t]) *((matrixSDT[s].getValue(d,t) + alpha) / (sumDST[d][s] + sumAlpha))*((matrixDS.getValue(d,s) + gammas[s]) / (sumDS[d] + sumGamma));
//							
//							//prob += this.Phi[s].getValue(w, t)*this.Theta[s].getValue(d, t)*this.Pi.getValue(d, s);
//						}
//					}
//					val = val + Math.log(prob);	
//				}
//			}	
//			
//			likelihood+= likelihood+val;
//			double tmp = Math.pow(2.0, -val/doc_lenght/log2);
//			System.out.println("The perplexity is:" + tmp);
//			if(!Double.isNaN(tmp) && !Double.isInfinite(tmp))
//				perplexity += tmp;
//			} // if part
//			
			
			// perplexity calculation
			//logLikelihood += Math.log(d.m_sstat[tid]/docSum * word_topic_sstat[tid][wid]/m_sstat[tid]);
					//perplexity += Math.pow(2.0, -loglikelihood/d.getTotalDocLength() / log2);
/*			int d = doc.getDocNo();
			double val = 0.0;
			int doc_lenght = 0;
			for (Sentence sentence : doc.getSentences()) {
				
				for (Word word : sentence.getWords()){
					double prob = 0.0; 
					int w = word.wordNo;
					doc_lenght++;
					for (int s = 0; s < numSenti; s++) {
						for (int t = 0; t < numTopics; t++) {
							prob += this.Phi[s].getValue(w, t)*((matrixSDT[s].getValue(d,t) + alpha) / (sumDST[d][s] + sumAlpha))*((matrixDS.getValue(d,s) + gammas[s]) / (sumDS[d] + sumGamma));
						}
					}
					val = val + Math.log(prob);	
				}
			}	
			
			double tmp = Math.pow(2.0, -val/doc_lenght / log2);
			out.println("The perplexity is:" + tmp);
			if(!Double.isNaN(tmp) && !Double.isInfinite(tmp))
				perplexity += tmp;
			} // if part
			*/
			
			// perplexity calculation
			//logLikelihood += Math.log(d.m_sstat[tid]/docSum * word_topic_sstat[tid][wid]/m_sstat[tid]);
			//perplexity += Math.pow(2.0, -loglikelihood/d.getTotalDocLength() / log2);
					
			int numWords = matrixSWT[0].getNumOfRow();
			Integer [] wordLexicons = new Integer[numWords];
			for (int w = 0; w < numWords; w++) {
					wordLexicons[w] = null;
					for (int s = 0; s < numSenti; s++) {
						if (sentiWordsList.get(s).contains(w)) wordLexicons[w] = s;
						}
				}
			
			int d = doc.getDocNo();
			double val = 0.0;
			int doc_lenght = 0;
			for (Sentence sentence : doc.getSentences()) {
				int s = sentence.getSenti();
				int t = sentence.getTopic();
				if(s==-1 || t==-1){
					System.out.println(sentence.getWordCnt().size());
					for (Word word : sentence.getWords()){
						System.out.print( wordList.get(word.getWordNo()) + " ");
					}
					System.out.println();
					continue;
				}
				
				double sentiSum = 0;
				for(int x=0; x<numSenti; x++)
					sentiSum+=(sumDST[d][x]);
				double probSenti = (sumDST[d][s])/(sentiSum);
				double topicSum = 0.0;
				for(int x=0; x<numTopics; x++)
					topicSum+=(matrixSDT[s].getValue(d, x));
				double probTopic = (matrixSDT[s].getValue(d, t))/(topicSum);
				System.out.println("Topic Sum:"+topicSum);
			
				for (Word word : sentence.getWords()){
					int w = word.wordNo;
					double beta;
					if (wordLexicons[w] == null) beta = betas[0];
					else if (wordLexicons[w] == s) beta = betas[1];
					else beta = betas[2];
					
					if(Phi[s].getValue(w, t)<0)
						System.err.print("OO error");

					//val=val+Math.log(Phi[s].getValue(w, t)*probSenti*probTopic);
					val = val + Math.log(Phi[s].getValue(w, t) * this.Theta[s].getValue(d, t) * this.Pi.getValue(d, s));
					if(Double.isNaN(val))
					{
						
						System.err.print("NaN error:");
						System.err.println(Phi[s].getValue(w, t) * this.Theta[s].getValue(d, t) * this.Pi.getValue(d, s));
						System.err.println(Math.log(Phi[s].getValue(w, t) * this.Theta[s].getValue(d, t) * this.Pi.getValue(d, s)));
						
					}
					doc_lenght++;
				}
			}	
			
			likelihood+=val;
			System.out.println("Likelihood"+ val);
			infoWriter.println("Likelihood"+ val);
			double tmp = Math.pow(2.0, -val/doc_lenght / log2);
			System.out.println("The perplexity is:" + tmp);
			infoWriter.println("The perplexity is:" + tmp);
			perplexity += tmp;
			} // if part
			
		}
		
		double pros_precision = (double)precision_recall[0][0]/(precision_recall[0][0] + precision_recall[1][0]);
		double cons_precision = (double)precision_recall[1][1]/(precision_recall[0][1] + precision_recall[1][1]);
		double pros_recall = (double)precision_recall[0][0]/(precision_recall[0][0] + precision_recall[0][1]);
		double cons_recall = (double)precision_recall[1][1]/(precision_recall[1][0] + precision_recall[1][1]);
		//System.out.println("pros_precision:"+pros_precision+" pros_recall:"+pros_recall);
		//System.out.println("cons_precision:"+cons_precision+" cons_recall:"+cons_recall);
		double pros_precision_sentence = (double)precision_recall_sentences[0][0]/(precision_recall_sentences[0][0] + precision_recall_sentences[1][0]);
		double cons_precision_sentence = (double)precision_recall_sentences[1][1]/(precision_recall_sentences[0][1] + precision_recall_sentences[1][1]);
		double pros_recall_sentence = (double)precision_recall_sentences[0][0]/(precision_recall_sentences[0][0] + precision_recall_sentences[0][1]);
		double cons_recall_sentence = (double)precision_recall_sentences[1][1]/(precision_recall_sentences[1][0] + precision_recall_sentences[1][1]);
		
		infoWriter.println("pros_precision_sentence:"+pros_precision_sentence+" pros_recall_sentence:"+pros_recall_sentence);
		infoWriter.println("cons_precision_sentence:"+cons_precision_sentence+" cons_recall_sentence:"+cons_recall_sentence);
//		
		
		System.out.println("pros_precision_sentence:"+pros_precision_sentence+" pros_recall_sentence:"+pros_recall_sentence);
		System.out.println("cons_precision_sentence:"+cons_precision_sentence+" cons_recall_sentence:"+cons_recall_sentence);
//		System.out.println("Word Level");
//		for(int i=0; i<2; i++)
//		{
//			for(int j=0; j<2; j++)
//			{
//				System.out.print(precision_recall[i][j]+",");
//			}
//			System.out.println();
//		}

		System.out.println("Sentence Level");
		infoWriter.println("Sentence Level");
		for(int i=0; i<2; i++){
			for(int j=0; j<2; j++){
				System.out.print(precision_recall_sentences[i][j]+",");
				infoWriter.print(precision_recall_sentences[i][j]+",");
			}
			System.out.println();
			infoWriter.println();
		}
		double pros_f1 = 2/(1/pros_precision + 1/pros_recall);
		double cons_f1 = 2/(1/cons_precision + 1/cons_recall);
//		
//		System.out.println("Word level F1 measure:pros:"+pros_f1+", cons:"+cons_f1);
		
		pros_f1 = 2/(1/pros_precision_sentence + 1/pros_recall_sentence);
		cons_f1 = 2/(1/cons_precision_sentence + 1/cons_recall_sentence);
		System.out.println("Sentence level F1 measure:pros:"+pros_f1+", cons:"+cons_f1);
		infoWriter.println("Sentence level F1 measure:pros:"+pros_f1+", cons:"+cons_f1);
		out.close();
		
		perplexity = perplexity/number_of_test_doc;
		System.out.println("perplexity:"+ perplexity);
		System.out.println("Likelihood:"+likelihood);
		
		infoWriter.println("perplexity:"+ perplexity);
		infoWriter.println("Likelihood:"+likelihood);
		
		System.out.println("Cannot predict:" + cannotPredict);	
		
		
	} 
	
	

	public void docsummary(int topic)
	{
		infoWriter.println("Doc summary for topic:"+topic);
		//System.out.println("Doc summary for topic:"+topic);
		for (OrderedDocument d : documents){
			infoWriter.println("Doc Number:"+d.getDocNo());
			if(d.isTestDoc()){
				for(int s=0; s<this.numSenti;s++){
					double max = 0.0;
					int index = -1;
					int sentence_index = 0;
					for (Sentence sentence : d.getSentences()) {
						if (sentence.getSenti() < 0 || sentence.getSenti() >= this.numSenti || sentence.getWords().size() > this.maxSentenceLength) {
							sentence_index++;
							continue;
						}
						double prod = 1.0;
						for (Word word : sentence.getWords()){
							int w = word.wordNo;
							prod = prod * this.Phi[s].getValue(w, topic)*Theta[s].getValue(d.getDocNo(), topic)*this.Pi.getValue(d.getDocNo(),s);
						}
						if(prod>max){
							max = prod;
							index = sentence_index ;
						}
						sentence_index++;
					}
					sentence_index = 0;
					for (Sentence sentence : d.getSentences()) {
						if(sentence_index==index){
							for (Word word : sentence.getWords()){
								System.out.print(" "+this.wordList.get(word.wordNo));
								infoWriter.print(" "+this.wordList.get(word.wordNo));
								
							}
							System.out.println();
							infoWriter.println();
						}
						sentence_index++;
					}
				}
			}
		}
		
		infoWriter.flush();
		infoWriter.close();
	}
	
	
	public void docSummary(String[] productList){
		
		
		
		
		for(String prodID : productList) {
			for(int i=0; i<this.numTopics; i++){
				MyPriorityQueue<_RankItem> stnQueue = new MyPriorityQueue<_RankItem>(25);//top three sentences per topic per product
				
				for (OrderedDocument d : documents){
					if(!d.isTestDoc()){
						if(d.getItemID().equalsIgnoreCase(prodID)) {
						int s = 0;
							for(Sentence sentence : d.getSentences()){
								double prob = 1.0;
								for (Word word : sentence.getWords()){
									int w = word.wordNo;
									prob = prob * this.Phi[s].getValue(w, i)*Theta[s].getValue(d.getDocNo(), i)*this.Pi.getValue(d.getDocNo(),s);
								}
								prob /= sentence.getWords().size();
								stnQueue.add(new _RankItem(sentence.getRawSentence(), prob));
							}
							
						
					}
					}
				}				
				System.out.format("Product: %s, Topic: %d\n", prodID, i);
				infoWriter.format("Product: %s, Topic: %d\n", prodID, i);
				int senNumber = 0;
				for(_RankItem it:stnQueue){
					System.out.format("%s\t%.3f\n", it.m_name, it.m_value);	
					infoWriter.format("%s\t%.3f\n", it.m_name, it.m_value);
					summaryWriter.format("%s,%d,%d,%s\n", prodID, i, senNumber, it.m_name);
					senNumber++;
				}			
			}
			
			
			infoWriter.write("Cons Topic\n");
			System.out.println("Cons Topic\n");
			for(int i=0; i<this.numTopics; i++){
				MyPriorityQueue<_RankItem> stnQueue = new MyPriorityQueue<_RankItem>(10);//top three sentences per topic per product
				
				for (OrderedDocument d : documents){
					if(!d.isTestDoc()){
						if(d.getItemID().equalsIgnoreCase(prodID)) {

							int s=1;
							for(Sentence sentence : d.getSentences()){
								double prob = 1.0;
								for (Word word : sentence.getWords()){
									int w = word.wordNo;
									prob = prob * this.Phi[s].getValue(w, i)*Theta[s].getValue(d.getDocNo(), i)*this.Pi.getValue(d.getDocNo(),s);
								}
								prob /= sentence.getWords().size();
								stnQueue.add(new _RankItem(sentence.getRawSentence(), prob));
							}
							
						
					}
					}
				}				
				System.out.format("Product: %s, Topic: %d\n", prodID, i);
				infoWriter.format("Product: %s, Topic: %d\n", prodID, i);
				int senNumber=0;
				for(_RankItem it:stnQueue){
					System.out.format("%s\t%.3f\n", it.m_name, it.m_value);	
					infoWriter.format("%s\t%.3f\n", it.m_name, it.m_value);	
					summaryWriter.format("%s,%d,%d,%s\n", prodID, i+numTopics, senNumber, it.m_name);
					senNumber++;
				}			
			}
			
			
		}
		infoWriter.flush();
		infoWriter.close();
		
		summaryWriter.flush();
		summaryWriter.close();
	}

	
	
	private void sampleForDoc(OrderedDocument currentDoc) {
		int docNo = currentDoc.getDocNo();
		for (Sentence sentence : currentDoc.getSentences()) {
			if (sentence.getSenti() == -1 || sentence.getWords().size() > this.maxSentenceLength) continue;
			
			// if training doc from newEgg not testDoc
			if(!currentDoc.isTestDoc() && (sentence.label==0 || sentence.label==1)){// sentences from newEgg and we know the label so no sampling
				
				Map<Word,Integer> wordCnt = sentence.getWordCnt();
				double sumProb = 0;
				
				int oldTopic = sentence.getTopic();
				int oldSenti = sentence.getSenti();
				
				matrixSDT[oldSenti].decValue(docNo, oldTopic);
				matrixDS.decValue(docNo, oldSenti);
				
				sumDST[docNo][oldSenti]--;
				sumDS[docNo]--;

				for (Word sWord : sentence.getWords()) {
					matrixSWT[oldSenti].decValue(sWord.wordNo, oldTopic);
					sumSTW[oldSenti][oldTopic]--;
				}
			
				// Sampling
				for (int si = 0; si < numSenti; si++) {
					boolean trim = false;
					
					// Fast Trimming
					for (Word sWord : wordCnt.keySet()) {
						SentiWord word = (SentiWord) sWord;
						if (word.lexicon != null && word.lexicon != si) {
							trim = true;
							break;
						}
					}
					if (trim) {
						for (int ti = 0; ti < numTopics; ti++)
							probTable[ti][si] = 0;
					}
					else {
						for (int ti = 0; ti < numTopics; ti++) {
							double beta0 = sumSTW[si][ti] + sumBeta[si];
							int m0 = 0;
							double expectTSW = 1;
							
							for (Word sWord : wordCnt.keySet()) {
								SentiWord word = (SentiWord) sWord;
								
								double beta;
								if (word.lexicon == null) beta = this.betas[0];
								else if (word.lexicon == si) beta = this.betas[1];
								else beta = this.betas[2];
								
								double betaw = matrixSWT[si].getValue(word.wordNo, ti) + beta;
		
								int cnt = wordCnt.get(word);
								for (int m = 0; m < cnt; m++) {
									expectTSW *= (betaw + m) / (beta0 + m0);
									m0++;
								}
							}

							probTable[ti][si] = (matrixSDT[si].getValue(docNo, ti) + this.alpha) / (sumDST[docNo][si] + this.sumAlpha)
							* (matrixDS.getValue(docNo, si) + this.gammas[si])
							* expectTSW;
		
							sumProb += probTable[ti][si];
						}
					}
				}

				int newTopic = 0, newSenti = 0;
				double randNo = Math.random() * sumProb;
				double tmpSumProb = 0;
				boolean found = false;
				for (int ti = 0; ti < numTopics; ti++) {
					for (int si = 0; si < numSenti; si++) {
						tmpSumProb += probTable[ti][si];
						if (randNo <= tmpSumProb) {
							newTopic = ti;
							newSenti = si;
							found = true;
						}
						if (found) break;
					}
					if (found) break;
				}
				
				// overide with original senti
				newSenti = sentence.label;
				sentence.setTopic(newTopic);
				sentence.setSenti(newSenti);
				
				for (Word sWord : sentence.getWords()) {
					SentiWord word = (SentiWord) sWord;
					word.setTopic(newTopic);
					word.setSentiment(newSenti);
					matrixSWT[newSenti].incValue(word.wordNo, newTopic);
					sumSTW[newSenti][newTopic]++;
				}
				matrixSDT[newSenti].incValue(docNo, newTopic);
				matrixDS.incValue(docNo, newSenti);
				
				sumDST[docNo][newSenti]++;
				sumDS[docNo]++;
			
				continue;
			}
			
			
			Map<Word,Integer> wordCnt = sentence.getWordCnt();
			double sumProb = 0;
			
			int oldTopic = sentence.getTopic();
			int oldSenti = sentence.getSenti();
			
			matrixSDT[oldSenti].decValue(docNo, oldTopic);
			matrixDS.decValue(docNo, oldSenti);
			
			sumDST[docNo][oldSenti]--;
			sumDS[docNo]--;

			for (Word sWord : sentence.getWords()) {
				matrixSWT[oldSenti].decValue(sWord.wordNo, oldTopic);
				sumSTW[oldSenti][oldTopic]--;
			}
		
			// Sampling
			for (int si = 0; si < numSenti; si++) {
				boolean trim = false;
				
				// Fast Trimming
				for (Word sWord : wordCnt.keySet()) {
					SentiWord word = (SentiWord) sWord;
					if (word.lexicon != null && word.lexicon != si) {
						trim = true;
						break;
					}
				}
				if (trim) {
					for (int ti = 0; ti < numTopics; ti++)
						probTable[ti][si] = 0;
				}
				else {
					for (int ti = 0; ti < numTopics; ti++) {
						double beta0 = sumSTW[si][ti] + sumBeta[si];
						int m0 = 0;
						double expectTSW = 1;
						
						for (Word sWord : wordCnt.keySet()) {
							SentiWord word = (SentiWord) sWord;
							
							double beta;
							if (word.lexicon == null) beta = this.betas[0];
							else if (word.lexicon == si) beta = this.betas[1];
							else beta = this.betas[2];
							
							double betaw = matrixSWT[si].getValue(word.wordNo, ti) + beta;
	
							int cnt = wordCnt.get(word);
							for (int m = 0; m < cnt; m++) {
								expectTSW *= (betaw + m) / (beta0 + m0);
								m0++;
							}
						}

						probTable[ti][si] = (matrixSDT[si].getValue(docNo, ti) + this.alpha) / (sumDST[docNo][si] + this.sumAlpha)
						* (matrixDS.getValue(docNo, si) + this.gammas[si])
						* expectTSW;
	
						sumProb += probTable[ti][si];
					}
				}
			}

			int newTopic = 0, newSenti = 0;
			double randNo = Math.random() * sumProb;
			double tmpSumProb = 0;
			boolean found = false;
			for (int ti = 0; ti < numTopics; ti++) {
				for (int si = 0; si < numSenti; si++) {
					tmpSumProb += probTable[ti][si];
					if (randNo <= tmpSumProb) {
						newTopic = ti;
						newSenti = si;
						found = true;
					}
					if (found) break;
				}
				if (found) break;
			}
			
			sentence.setTopic(newTopic);
			sentence.setSenti(newSenti);
			
			for (Word sWord : sentence.getWords()) {
				SentiWord word = (SentiWord) sWord;
				word.setTopic(newTopic);
				word.setSentiment(newSenti);
				matrixSWT[newSenti].incValue(word.wordNo, newTopic);
				sumSTW[newSenti][newTopic]++;
			}
			matrixSDT[newSenti].incValue(docNo, newTopic);
			matrixDS.incValue(docNo, newSenti);
			
			sumDST[docNo][newSenti]++;
			sumDS[docNo]++;
		}
	}
	
	public void generateTmpOutputFiles(String inputDir, String outputDir, int interval) throws Exception {
		if (inputDir == null || outputDir == null) throw new Exception("Should specify the input and output dirs for tmp output files");
		if (interval <= 0) throw new Exception("The interval of writing tmp output files should be greater than 0");
		this.inputDir = inputDir;
		this.outputDir = outputDir;
		this.intvalTmpOutput = interval;
	}
	
	public void generateOutputFiles (String dir) throws Exception {
		String prefix = "STO2-T"+numTopics+"-S"+numSenti+"("+sentiWordsList.size()+")-A"+alpha+"-B"+betas[0];
		for (int i = 1; i < betas.length; i++) prefix += ","+betas[i];
		prefix += "-G"+gammas[0];
		for (int i = 1; i < numSenti; i++) prefix += ","+gammas[i];
		prefix += "-I"+numRealIterations;
		
		// Phi
		System.out.println("Writing Phi...");
		PrintWriter out = new PrintWriter(new FileWriter(new File(dir + "/" + prefix + "-Phi.csv")));
		for (int s = 0; s < this.numSenti; s++)
			for (int t = 0; t < this.numTopics; t++)
				out.print(",S"+s+"-T"+t);
		out.println();
		for (int w = 0; w < this.wordList.size(); w++) {
			out.print(this.wordList.get(w));
			for (int s = 0; s < this.numSenti; s++) {
				for (int t = 0; t < this.numTopics; t++) {
					out.print(","+this.Phi[s].getValue(w, t));
				}
			}
			out.println();
		}
		out.close();

		// Theta
		System.out.println("Writing Theta...");
		out = new PrintWriter(new FileWriter(new File(dir + "/" + prefix + "-Theta.csv")));
		for (int s = 0; s < this.numSenti; s++)
			for (int t = 0; t < this.numTopics; t++)
				out.print("S"+s+"-T"+t+",");
		out.println();
		for (int d = 0; d < this.numDocuments; d++) {
			for (int s = 0; s < this.numSenti; s++) {
				for (int t = 0; t < this.numTopics; t++) {
					out.print(this.Theta[s].getValue(d, t)+",");
				}
			}
			out.println();
		}
		out.close();
		
		// Pi
		System.out.println("Writing Pi...");
		this.Pi.writeMatrixToCSVFile(dir + "/" + prefix + "-Pi.csv");
		
		// Most probable words
		System.out.println("Writing the most probable words...");
		out = new PrintWriter(new FileWriter(new File(dir + "/" + prefix + "-ProbWords.csv")));
		for (int s = 0; s < this.numSenti; s++)
			for (int t = 0; t < this.numTopics; t++)
				out.print("S"+s+"-T"+t+",");
		out.println();
		int [][][] wordIndices = new int[this.numSenti][this.numTopics][this.numProbWords];
		for (int s = 0; s < this.numSenti; s++) {
			for (int t = 0; t < this.numTopics; t++) {
				Vector<Integer> sortedIndexList = this.Phi[s].getSortedColIndex(t, this.numProbWords);
				for (int w = 0; w < sortedIndexList.size(); w++)
					wordIndices[s][t][w] = sortedIndexList.get(w);
			}
		}
		for (int w = 0; w < this.numProbWords; w++) {
			for (int s = 0; s < this.numSenti; s++) {
				for (int t = 0; t < this.numTopics; t++) {
					int index = wordIndices[s][t][w];
					out.print(this.wordList.get(index)+" ("+String.format("%.3f", Phi[s].getValue(index,t))+"),");
				}
			}
			out.println();
		}
		out.close();


		// Intrusion words
		HashMap<String, ArrayList<String>> list = new HashMap<String, ArrayList<String>>();
		System.out.println("Writing the intrusion list words...");
		out = new PrintWriter(new FileWriter(new File(this.wordIntrusionFilePath)));
		for (int s = 0; s < this.numSenti; s++)
			for (int t = 0; t < this.numTopics; t++)
				out.print("S"+s+"-T"+t+",");
		out.println();
	
		for (int w = 0; w < 5; w++) {
			for (int s = 0; s < this.numSenti; s++) {
				for (int t = 0; t < this.numTopics; t++) {
					int index = wordIndices[s][t][w];
					out.print(this.wordList.get(index)+",");
					String tmp = ""+s+t;
					if(list.containsKey(tmp)){
						list.get(tmp).add(this.wordList.get(index));
					}
					else{
						ArrayList<String> words = new ArrayList<String>();
						words.add(this.wordList.get(index));
						list.put(tmp, words);
					}
				}
			}
			out.println();
		}
		
		// inter topic intrusion word index
		Random r = new Random();
		int w = 90 + r.nextInt(100);
		for (int s = 0; s < this.numSenti; s++) {
			for (int t = 0; t < this.numTopics; t++) {
				int index = wordIndices[s][t][w];
				out.print(this.wordList.get(index)+",");
			}
		}
		out.println();
		
		// intra topic intrusion word index
		
		for (int s = 0; s < this.numSenti; s++) {
			for (int t = 0; t < this.numTopics; t++) {
				//select next topic
				int nextTopic = (t + 1)%this.numTopics;
				
			
				String key = ""+s+t;
				ArrayList<String> words = list.get(key);
				
				
				int wordIndex = 0;
				int index = wordIndices[s][nextTopic][wordIndex];
				
				
				while(true){
					String word = this.wordList.get(index);
					if(!words.contains(word)){
						out.print(word+",");
						break;
					}
					wordIndex++;
					index = wordIndices[s][nextTopic][wordIndex];
				
				}
				
				
			}
		}
		out.println();
		
		out.close();

		
		
		
		/*
		// Result reviews
		System.out.println("Visualizing reviews...");
		String [] sentiColors = {"green","red","black"};
		out = new PrintWriter(new FileWriter(new File(dir + "/" + prefix + "-VisReviews.html")));
		for (OrderedDocument doc : this.documents) {
			out.println("<h3>Document "+doc.getDocNo()+"</h3>");
			for (Sentence sentence : doc.getSentences()) {
				if (sentence.getSenti() < 0 || sentence.getSenti() >= this.numSenti || sentence.getWords().size() > this.maxSentenceLength) continue;
				out.print("<p style=\"color:"+sentiColors[sentence.getSenti()]+";\">T"+sentence.getTopic()+":");
				for (Word word : sentence.getWords()) out.print(" "+this.wordList.get(word.wordNo));
				out.println("</p>");
			}
		}
		out.close();
		
		// Sentence probabilities
		System.out.println("Calculating sentence probabilities...");
		out = new PrintWriter(new FileWriter(new File(dir + "/" + prefix + "-SentenceProb.csv")));
		out.print("Document,Sentence,Length");
		for (int s = 0; s < this.numSenti; s++)
			for (int t = 0; t < this.numTopics; t++)
				out.print(",S"+s+"-T"+t);
		out.println();
		for (int d = 0; d < this.documents.size(); d++) {
			OrderedDocument doc = this.documents.get(d);
			for (int sen = 0; sen < doc.getSentences().size(); sen++) {
				Sentence sentence = doc.getSentences().get(sen);
				if (sentence.numSenti > 1 || sentence.getWords().size() > 50) continue;
				if (sentence.getWords().size() == 0) throw new Exception("WHAT???");
				out.print(d+",\"");
				for (Word word : sentence.getWords())
					out.print(this.wordList.get(word.wordNo)+" ");
				out.print("\","+sentence.getWords().size());
				
				double [][] prod = new double[this.numSenti][this.numTopics];
				double sum = 0;
				for (int s = 0; s < this.numSenti; s++) {
					for (int t = 0; t < this.numTopics; t++) {
						prod[s][t]  = 1;
						for (Word word : sentence.getWords()) prod[s][t] *= this.Phi[s].getValue(word.wordNo, t);
						sum += prod[s][t];
					}
				}
				for (int s = 0; s < this.numSenti; s++) {
					for (int t = 0; t < this.numTopics; t++) {
						out.print("," + (prod[s][t] / sum));
					}
				}
				out.println();
			}
		}
		out.close();
		
		// Sentiment lexicon words distribution
		System.out.println("Calculating sentiment lexicon words distributions...");
		out = new PrintWriter(new FileWriter(new File(dir + "/" + prefix + "-SentiLexiWords.csv")));
		for (Set<Integer> sentiWords : this.sentiWordsList) {
			for (int wordNo : sentiWords) {
				if (wordNo < 0 || wordNo >= this.wordList.size()) continue;
				out.print(this.wordList.get(wordNo));
				for (int s = 0; s < numSenti; s++) {
					int sum = 0;
					for (int t = 0; t < numTopics; t++) sum += matrixSWT[s].getValue(wordNo, t);
					out.print(","+sum);
				}
				out.println();
			}
			out.println();
		}
		out.close();
		*/
	}
}
