import com.wcohen.ss.JaroWinkler;
import org.apache.jena.query.*;
import org.apache.jena.rdf.model.*;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Random;

public class ValidationTool {

    public static void main(String[] args) throws IOException {


        //initialization read files
        Model model0 = ModelFactory.createDefaultModel();
        model0.read("IIMB_LARGE/000/onto.owl");

        Model model1 = ModelFactory.createDefaultModel();
        model1.read("IIMB_LARGE/001/onto.owl");

        List<String> functionalProperties = Files.readAllLines(Paths.get("fps.txt"));
        Model modelPairs = ModelFactory.createDefaultModel();
        modelPairs.read("IIMB_LARGE/001/refalign.rdf");


        String queryString =
                //"PREFIX xmlns: <http://knowledgeweb.semanticweb.org/heterogeneity/alignment#>"+
                "PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>"+
                "PREFIX align: <http://knowledgeweb.semanticweb.org/heterogeneity/alignment#>"+
                "PREFIX xsd: <http://www.w3.org/2001/XMLSchema#>" +
                "SELECT ?x ?y WHERE {"+
                "?entity align:entity1 ?x ."+
                "?entity align:entity2 ?y"+
                "}";
        Query query = QueryFactory.create(queryString);
        QueryExecution qexec = QueryExecutionFactory.create(query, modelPairs);
        List<Pair> correctPairs = new ArrayList<>();
        try {
            ResultSet results = qexec.execSelect();
            while (results.hasNext()){
                QuerySolution soln = results.nextSolution();
                String entity1 = soln.get("x").toString();
                String entity2 = soln.get("y").toString();
                Pair pair = new Pair(entity1, entity2, true);
                correctPairs.add(pair);
            }
        } finally {
            qexec.close();
        }
        
        System.out.println("CorrectPairs: " + correctPairs.size());
        //generate random erroneous sameAs links
        List<Pair> incorrectPairs = new ArrayList<>();
        int numberOfErroneousLinks = 1416; //parameter that can be modified
        Random random = new Random();
        for (int i=0; i<numberOfErroneousLinks; i++) {
            int idx1 = random.nextInt(correctPairs.size());
            String entity1 = correctPairs.get(idx1).getEntity1();
            //make sure that the second random number is different
            int idx2 = 0;
            do{
                idx2 = random.nextInt(correctPairs.size());
            }while(idx2==idx1);
            String entity2 = correctPairs.get(idx2).getEntity2();
            incorrectPairs.add(new Pair(entity1, entity2, false));
        }

 
        // combine incorrect pairs with correct pairs and shuffle them
        List<Pair> PairsSet = new ArrayList<>();
        PairsSet.addAll(correctPairs);
        PairsSet.addAll(incorrectPairs);
        Collections.shuffle(PairsSet);
        
//        List<Double> listThreshold = Arrays.asList(0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 
//                                  0.40, 0.45, 0.50, 0.55, 0.60, 0.65,
//                                  0.70, 0.75, 0.80, 0.85, 0.90, 0.95);
//        List<Double> listAccuracy = new ArrayList<Double>();
//        List<Double> listPrecision = new ArrayList<Double>();
//        List<Double> listRecall = new ArrayList<Double>();
//        List<Double> listF1 = new ArrayList<Double>();

        
//        for (int i=0; i<listThreshold.size();i++) {
            double similarityThreshold = 0.55; //parameter modifiable TODO:参数可以多试试
            double truePositive = 0;
            double falsePositive = 0;
            double trueNegative = 0;
            double falseNegative = 0;

            for (Pair pair : PairsSet) {
                Resource resource1 = model0.getResource(pair.getEntity1());
                pair.setPredicted(true);
                StmtIterator it1 = resource1.listProperties();
                while (it1.hasNext()) {
                    Statement st1 = it1.nextStatement();
                    if (functionalProperties.contains(st1.getPredicate().toString())) {
                        String str1 = st1.getObject().toString();
                        Resource resource2 = model1.getResource(pair.getEntity2());
                        StmtIterator it2 = resource2.listProperties();
                        while (it2.hasNext()) {
                            Statement st2 = it2.nextStatement();
                            if (st2.getPredicate().toString().equals(st1.getPredicate().toString())) {
                                String str2 = st2.getObject().toString();
                                
                                // change 1984-11-21 to 11/21/84
//                                String regex = "([0-9]+)(-)([0-9]+)(-)([0-9]+)";
//                                Pattern pattern = Pattern.compile(regex);
//                                Matcher m = pattern.matcher(str1);
//                                boolean dateFlag = m.find();
//                                if(dateFlag){
//                                	if(m.groupCount()==5 & m.group(1).length()==4) {
//                                		str1 = m.group(3)+"/"+m.group(5)+"/"+m.group(1).substring(2,4);
//                                	    //System.out.println("Formalised str2: "+str1);
//                                	    }
//
//                            		}

                                //TODO: strings extracted can be normalized before this, such as different data formats, "Male" to "M", "Female" to "F", 多余的标点符号了， etc.
                                //TODO：举个例子，像born_in这个关系，因为后面跟的是实体，在000里面是http://blabla/Paris,在001里面就是http://blabla/item36259425之类，虽然实际上是一个东西，但字符串比较结果会不一样，此处可改进，不好改进就写近报告里面的待改进部分
                                JaroWinkler jaroWinkler = new JaroWinkler();
                                double similarity = jaroWinkler.score(str1, str2);
                                if (similarity < similarityThreshold) {
                                    pair.setPredicted(false);
                                    pair.setEntity1PropertyValue(str1);
                                    pair.setEntity2PropertyValue(str2);
                                    pair.setProperty(st1.getPredicate().toString());
                                }
                            }
                        }
                    }
                }
                if (pair.isValid()==true & pair.getPredicted() == true) {
                    truePositive += 1;
                } else if (pair.isValid()==false & pair.getPredicted()==false) {
                	trueNegative += 1;
    			}else if (pair.isValid()==true & pair.getPredicted() == false) {
    				falseNegative += 1;
    				System.out.println("False Negative: ");
    				System.out.println("Different property: " + pair.getProperty());
    				System.out.println("Entity1: " + pair.getProperty1());
    				System.out.println("Entity2: " + pair.getProperty2());    				
    				
    			}else {
    				falsePositive += 1;
    			}
            }


            //TODO：不知道接下来三行写的对不对
            double accuracy = (truePositive + trueNegative)/ PairsSet.size();
            double precision = truePositive / (truePositive + falsePositive);
            double recall = truePositive / (truePositive + falseNegative);
            double f1 = 2 * ((precision * recall) / (precision + recall));
            
            
//            listAccuracy.add(accuracy);
//            listPrecision.add(precision);
//            listRecall.add(recall);
//            listF1.add(f1);
//
//
//        	
//        }
//        System.out.println("Accuracy: " + listAccuracy);
//        System.out.println("Precision: " + listPrecision);
//        System.out.println("Recall: " + listRecall);
//        System.out.println("f1: " + listF1);

          System.out.println("Accuracy: " + accuracy);
          System.out.println("Precision: " + precision);
          System.out.println("Recall: " + recall);
          System.out.println("f1: " + f1);
    }
}
