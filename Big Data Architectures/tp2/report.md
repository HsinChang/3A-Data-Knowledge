# Big Data Architecture Lab 2
### ZHANG Xin

## Task 1: data import
### 1. import the files
Command:
```sql
LOAD CSV WITH HEADERS FROM "file:///boston-crime-incident-reports-10k.csv" AS row
CREATE (:Reports {incidentNumber: row.INCIDENT_NUMBER, OffenceCode: row.OFFENSE_CODE, offenseCodeGroup: row.OFFENSE_CODE_GROUP});
```
Result:
```
Added 9999 labels, created 9999 nodes, set 29997 properties, completed after 649 ms.
```
Commande:
```sql
LOAD CSV WITH HEADERS FROM "file:///boston-offense-codes-lookup.csv" AS row
CREATE (:Lookup {code: row.CODE, name: row.NAME});
```
Result:
```
Added 576 labels, created 576 nodes, set 1152 properties, completed after 72 ms.
```
### 2. Model data as a property graph
Command:
```sql

```