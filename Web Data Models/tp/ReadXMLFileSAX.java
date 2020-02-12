import javax.xml.parsers.SAXParser;
import javax.xml.parsers.SAXParserFactory;
import org.xml.sax.Attributes;
import org.xml.sax.SAXException;
import org.xml.sax.helpers.DefaultHandler;
import java.net.URLEncoder;
import java.io.*;
import org.xml.sax.*;
import java.net.URL;

public class ReadXMLFileSAX {

   public static void main(String argv[]) {

    try {

	SAXParserFactory factory = SAXParserFactory.newInstance();
	SAXParser saxParser = factory.newSAXParser();
	SAXParser sp1 = factory.newSAXParser();

	DefaultHandler handler1 = new DefaultHandler() {

	boolean ext = false;
	String url = new String();

	public void startElement(String uri, String localName,String qName,
                Attributes attributes) throws SAXException {

		if(qName.equalsIgnoreCase("extract")){
			ext = true;
		}



		super.startElement(uri, localName, qName, attributes);
	}

	public void endElement(String uri, String localName,
		String qName) throws SAXException {

		super.endElement(uri, localName, qName);
	}

	public void characters(char ch[], int start, int length) throws SAXException {

		if (ext) {
			System.out.println("Description: " + new String(ch, start, length));
			ext = false;
		}

	}

     };

	DefaultHandler handler = new DefaultHandler() {

	boolean bname = false;
	boolean ext = false;
	String url = new String();

	public void startElement(String uri, String localName,String qName,
                Attributes attributes) throws SAXException {

		if(qName.equalsIgnoreCase("tag")) {
			String att = attributes.getValue("k");//.equalsIgnoreCase("name")
			if(att.equalsIgnoreCase("name")){
				bname = true;
				String val = attributes.getValue("v");
				System.out.println("Name: " + val);
				try{
				String url = "https://en.wikipedia.org/w/api.php?format=xml&action=query&prop=extracts&exintro=&explaintext=&titles=" + URLEncoder.encode(val, "UTF-8");
				sp1.parse(new InputSource(new URL(url).openStream()), handler1);
			} catch(Exception e){
				e.printStackTrace();
			}
			}
		}

		super.startElement(uri, localName, qName, attributes);
	}

	public void endElement(String uri, String localName,
		String qName) throws SAXException {

		super.endElement(uri, localName, qName);
	}

	public void characters(char ch[], int start, int length) throws SAXException {

		if (bname) {
			//System.out.println("Name : " + new String(ch, start, length));
			bname = false;
		}

	}

     };

       saxParser.parse("map.osm", handler);

     } catch (Exception e) {
       e.printStackTrace();
     }

   }

}