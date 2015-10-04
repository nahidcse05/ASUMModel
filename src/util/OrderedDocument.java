package util;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;
import java.util.List;
import java.util.StringTokenizer;
import java.util.Vector;

public class OrderedDocument extends Document {

	private Vector<Sentence> sentences;
	
	public OrderedDocument() {
		super();
		sentences = new Vector<Sentence>();
	}
	
	public void addWord(Word word) {
		super.addWord(word);
		sentences.lastElement().addWord(word);
	}
	
	public void addWord(int wordNo) {
		addWord(new Word(wordNo));
	}
	
	public void addSentence(Sentence sentence) {
		sentences.add(sentence);
		for (Word word : sentence.getWords())
			words.add(word);
	}
	
	public Vector<Sentence> getSentences() {
		return sentences;
	}
	
	public static Vector<OrderedDocument> instantiateOrderedDocuments (String path, List<String> authors, List<String> authorList) throws Exception {
		Vector<OrderedDocument> documents = new Vector<OrderedDocument>();
		BufferedReader wordDocFile = new BufferedReader(new FileReader(new File(path)));
		
		int docCount=0;
		String line;
		while(true){
			
			line = wordDocFile.readLine();
			if(line == null) break;
			StringTokenizer st = new StringTokenizer(line);
			int numSentences = Integer.valueOf(st.nextToken());
			
			//reading the itemCategory
			line = wordDocFile.readLine();
			String itemID = line;
			System.out.println("i:"+itemID);
			if(line == null) break;
			
			
			
			OrderedDocument currentDoc = new OrderedDocument();			
			currentDoc.setDocNo(docCount++);
			currentDoc.setItemID(itemID);
			
			if (numSentences<0) // it is negative means it is for testing
			{
				currentDoc.setTestDoc(true);
				numSentences = Math.abs(numSentences);
			}
			
			for (int s = 0; s < numSentences; s++) {
				Sentence sentence = new Sentence();
				line = wordDocFile.readLine();
				if(line==null)
					continue;
				//System.out.println(docCount+" "+line);
				st = new StringTokenizer(line);
				
				
				//this part open when our code run
				sentence.label = Integer.valueOf(st.nextToken());
				if(sentence.label==-1){
					sentence.label = 0;
				}
				else if(sentence.label==-2){
				 sentence.label = 1;
				}
				else if(sentence.label==-3){
					 sentence.label = 2; // neutral from Amazon
				}
				while(st.hasMoreElements()){
					int wordNo = Integer.valueOf(st.nextToken());
					sentence.addWord(new SentiWord(wordNo));
				}
				
				// get the Raw sentence also
				line = wordDocFile.readLine();
				sentence.setRawSentence(line);
				
				currentDoc.addSentence(sentence);
			}
			
			if (authors != null) {
				int author = Integer.valueOf(authorList.indexOf(authors.get(currentDoc.getDocNo())) );
				currentDoc.setAuthor(author);
			}
			
			documents.add(currentDoc);
		}
		wordDocFile.close();
		
		return documents;
	}
	
}
