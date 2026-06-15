import sys
import os
from rdkit import Chem
from src.api.inference import ModelInference

def main():
    engine = ModelInference.get_instance()
    engine.load()
    
    # We will test compounds known to be extremely biologically active across multiple pathways:
    # 1. Polycyclic aromatic hydrocarbons and dioxins
    # 2. Steroids and active nuclear receptor agonists
    # 3. Metal complexes and known cell poisons
    
    test_compounds = [
        # Dioxin (TCDD) - extreme aryl hydrocarbon receptor agonist and general toxin
        ("TCDD (Dioxin)", "Clc1cc2Oc3cc(Cl)c(Cl)cc3Oc2cc1Cl"),
        # Diethylstilbestrol (DES) - extremely potent endocrine disruptor
        ("Diethylstilbestrol", "CCC(=C(CC)c1ccc(O)cc1)c1ccc(O)cc1"),
        # Benzo[a]pyrene - highly carcinogenic and mutagenic polycyclic aromatic hydrocarbon
        ("Benzo[a]pyrene", "c1ccc2c(c1)ccc3c2ccc4c5ccccc5ccc34"),
        # Estradiol - extremely high nuclear receptor activity
        ("Estradiol", "CC12CCC3C(C1CCC2O)CCC4=C3C=CC(=C4)O"),
        # Digoxin - highly active cardiac glycoside and cell pathway disruptor
        ("Digoxin", "CC1C(C(CC(O1)OC2C(OC(CC2O)OC3C(OC(CC3O)OC4CCC5(C(C4)CCC6C5CC(C7(C6(CCC7C8=CC(=O)OC8)O)C)O)C)C)C)O)O"),
        # BPA (Bisphenol A)
        ("Bisphenol A", "CC(C)(c1ccc(O)cc1)c1ccc(O)cc1"),
        # Rotenone
        ("Rotenone", "COC1=C(C2=C(C=C1)OC3C(C2=O)C4=C(C=C5C(=C4)OC(C5)(C)C=C)OC3)OC"),
        # Heavy metal complex ligand: e.g. Thiram (highly toxic fungicide)
        ("Thiram", "CN(C)C(=S)SSC(=S)N(C)C"),
        # Kepone (Chlordecone)
        ("Kepone", "ClC12C3(Cl)C4(Cl)C5(Cl)C(Cl)(Cl)C1(Cl)C5(Cl)C2(Cl)C3(Cl)C4=O"),
        # Endosulfan
        ("Endosulfan", "ClC1=C(Cl)C2(Cl)C3COSOCO3C1(Cl)C2(Cl)Cl"),
        # DDT
        ("DDT", "Clc1ccc(cc1)C(C(Cl)(Cl)Cl)c2ccc(Cl)cc2"),
        # Tamoxifen
        ("Tamoxifen", "CCN(C)CCOc1ccc(cc1)C(=C(CC)c2ccccc2)c3ccccc3"),
        # Tributyltin hydride / organotins (very high toxicity)
        ("Tributyltin hydride group", "CCCC[Sn](CCCC)CCCC"),
        # Hexachlorobenzene
        ("Hexachlorobenzene", "c1(c(c(c(c(c1Cl)Cl)Cl)Cl)Cl)Cl"),
    ]
    
    # We will also try to search in the Tox21 dataset labels if we can find any recorded in our SQLite database
    # that already have multiple labels. Let's do that if nothing in test_compounds achieves 12.
    
    print("Evaluating test compounds...")
    found_any = False
    
    for name, smiles in test_compounds:
        res = engine.predict_single(smiles)
        probs = [t['probability'] for t in res['predictions'].values()]
        
        # Test thresholds
        for th in [0.20, 0.15, 0.10, 0.05]:
            flagged = [k for k, v in res['predictions'].items() if v['probability'] >= th]
            if len(flagged) == 12:
                print(f"\nFOUND ONE at threshold {th:.2f}!")
                print(f"Name: {name}")
                print(f"SMILES: {smiles}")
                print("Task probabilities:")
                for k, v in res['predictions'].items():
                    print(f"  - {k}: {v['probability']:.4f} ({v['task_description']})")
                found_any = True
                return
                
    if not found_any:
        print("\nChecking SQLite database molecules for high probability matches...")
        import sqlite3
        import json
        if os.path.exists("graphmol.db"):
            conn = sqlite3.connect("graphmol.db")
            c = conn.cursor()
            rows = c.execute("SELECT input_smiles, predictions FROM prediction_requests WHERE is_valid_smiles = 1").fetchall()
            for smi, preds_json in rows:
                if not preds_json:
                    continue
                preds = json.loads(preds_json)
                probs = [v['probability'] for v in preds.values()]
                for th in [0.20, 0.15, 0.10, 0.05]:
                    flagged = [k for k, v in preds.items() if v['probability'] >= th]
                    if len(flagged) == 12:
                        print(f"\nFOUND ONE in DB history at threshold {th:.2f}!")
                        print(f"SMILES: {smi}")
                        print("Task probabilities:")
                        for k, v in preds.items():
                            print(f"  - {k}: {v['probability']:.4f}")
                        return
            print("No 12-flagged molecules found in DB history either.")

if __name__ == "__main__":
    main()
