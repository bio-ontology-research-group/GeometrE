<table>
  <tr>
    <!-- Titles -->
    <td align="center"><strong>With normal loss</strong></td>
    <td align="center"><strong>With transitive loss</strong></td>
  </tr>
  <tr>
    <!-- GIFs -->
    <td align="center"><img src="elbe.gif" alt="Normal loss" width="400"/></td>
    <td align="center"><img src="elbe2.gif" alt="Transitive loss" width="400"/></td>
  </tr>
</table>


## Datasets preparation

### Link prediction task

Data is already preprocessed. We show the scripts we used to process it:
For each dataset (WN18RR, FB15k-237), run the following scripts:

```
cd transEL/src/
python saturate_graph.py -ds [wn18rr|fb15k237]
python preprocess_kg.py -ds [wn18rr|fb15k237]
python add_rbox.py -ds [wn18rr|fb15k237]
```

### 
