public class Pair {
    private String entity1;
    private String entity2;
    private boolean isValid;
    private boolean predicted;
    private String property;
    private String entity1PropertyValue;
    private String entity2PropertyValue;
    

    public Pair(String left, String right, boolean b) {
        this.entity1 = left;
        this.entity2 = right;
        isValid = b;
        property = "";
        entity1PropertyValue = "";
        entity1PropertyValue = "";
        
    }

    public boolean isValid() {
        return isValid;
    }

    public void setValid(boolean valid) {
        isValid = valid;
    }

    public void setPredicted(boolean predict){
        predicted = predict;
    }
    
    public boolean getPredicted(){
        return predicted;
    }
    
    public void setProperty(String str){
    	property = str;
    }  
    
    public String getProperty(){
    	return property;
    }  
    
    public void setEntity1PropertyValue(String str1){
    	entity1PropertyValue = str1;
    }
    
    public void setEntity2PropertyValue(String str2){
    	entity2PropertyValue = str2;
    }
    
    public String getProperty1(){
        return entity1PropertyValue;
    }
    public String getProperty2(){
        return entity2PropertyValue;
    }

    public String getEntity1() {
        return entity1;
    }

    public void setEntity1(String entity1) {
        this.entity1 = entity1;
    }

    public String getEntity2() {
        return entity2;
    }

    public void setEntity2(String entity2) {
        this.entity2 = entity2;
    }
}