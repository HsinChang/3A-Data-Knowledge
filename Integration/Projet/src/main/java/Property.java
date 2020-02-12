import java.util.*;

class Property {

    private float functionalDegree;

    private String name;

    private Map<String, Integer> subjectCount;

    public Property(String name){
        this.name = name;
        this.subjectCount = new HashMap<String, Integer>();
        this.functionalDegree = 0;
    }

    public void addSubject(String subject) {
        Integer count = subjectCount.getOrDefault(subject, 0);  
        subjectCount.put(subject, count + 1);
        //recalculate functional degree
        Iterator it = subjectCount.entrySet().iterator();
        float totalSubjects = 0;
        float totalTriples = 0;
        while (it.hasNext()) {
            totalSubjects += 1;
            Map.Entry pair = (Map.Entry) it.next();
            int value = (int) pair.getValue();
            totalTriples += value;
        }
        functionalDegree = totalSubjects / totalTriples;
    }

    public float getFunctionalDegree() {
        return this.functionalDegree;
    }
}