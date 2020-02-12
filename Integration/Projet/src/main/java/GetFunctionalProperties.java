import org.apache.jena.rdf.model.Model;
import org.apache.jena.rdf.model.ModelFactory;
import org.apache.jena.rdf.model.Statement;
import org.apache.jena.rdf.model.StmtIterator;

import java.io.FileWriter;
import java.io.IOException;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

public class GetFunctionalProperties {

    public static void main(String[] args) throws IOException {
        double threshold = 0.98; // The threshold of functional degree to filter out desired properties.
        //get model 000.ox
        Model model = ModelFactory.createDefaultModel();
        model.read("IIMB_LARGE/000/onto.owl");

        // extract all properties
        //TODO: better to use SPARQL to retrieve the data, I have no time to test, follow this link: https://jena.apache.org/documentation/query/app_api.html and https://www.youtube.com/watch?v=nUdHneViLp4
        //TODO: A really bad practice to iterate over all statements, should use SPARQL. SPARQL can also eliminate the "http://oaei.ontologymatching.org/2010/IIMBTBOX/" part
        StmtIterator itr = model.listStatements();
        Map<String, Property> properties = new HashMap<>();
        while (itr.hasNext()){
            Statement st = itr.nextStatement();

            String sub = st.getSubject().toString();
            String pred = st.getPredicate().toString();

            if(pred.contains("IIMBTBOX") || pred.contains("IIMB2010")){
                Property property = properties.get(pred);
                if (property == null) {
                    properties.put(pred, new Property(pred));
                } else {
                    property.addSubject(sub);
                }
            }
        }



        //filter out functional properties
        List<String> functionalProperties = new ArrayList<>();
        properties.forEach((key, prop)-> {
        	System.out.println("key: " + key);
        	System.out.println("prop: " + prop);
            if (prop.getFunctionalDegree() > threshold) {
                functionalProperties.add(key);
            }
        });

        //TODO: some of the properties filtered out by this degree may not be practical(i.e. those with numerical values), needs to verify manually. In this case, just get rid of them here. functionProperties is just a list of strings.
        //TODO: cases like sub-properties and inversOf not considered
        //write functional properties to a file:
        FileWriter writer = new FileWriter("fps.txt");
        for(String str: functionalProperties) {
            writer.write(str + System.lineSeparator());
        }
        writer.close();

    }
}
