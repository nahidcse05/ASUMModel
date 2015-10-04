package util;

import java.util.TreeMap;
import java.util.Vector;

public class Sentence {

	private Vector<Word> words;
	private TreeMap<Word,Integer> wordCnt;  // Somehow HashMap doesn't work correctly
	private int topic=-1;
	private int senti=-1;
	public int numSenti;
	public int label = -1; // 0 = pros, 1 = cons and 2 = from Amazon neutral, -1 means 
	public String m_rawSource;
	
	public Sentence() {
		words = new Vector<Word>();
		wordCnt = new TreeMap<Word,Integer>();
	}
	
	public void setRawSentence(String sen){
		m_rawSource = sen;
	}
	
	public String getRawSentence(){
		return m_rawSource;
	}
	
	public void addWord(Word word) {
		words.add(word);
		Integer cnt = wordCnt.get(word);
		if (cnt == null) wordCnt.put(word, 1);
		else wordCnt.put(word, cnt+1);
	}
	
	public Vector<Word> getWords() {
		return words;
	}
	
	public TreeMap<Word,Integer> getWordCnt() {
		return wordCnt;
	}
	
	public int getTopic() {
		return topic;
	}
	
	public int getSenti() {
		return senti;
	}
	
	public void setTopic(int topic) {
		this.topic = topic;
	}
	
	public void setSenti(int senti) {
		this.senti = senti;
	}
}
