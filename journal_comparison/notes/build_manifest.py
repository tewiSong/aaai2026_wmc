"""Validate the downloaded PDF files and record provenance; no network calls."""
from pathlib import Path
import hashlib
import json
from pypdf import PdfReader

ROOT = Path(__file__).resolve().parents[1]
RECORDS = [
    dict(id="J1", journal="JAIR", year=2011,
         title="Exploiting Structure in Weighted Model Counting Approaches to Probabilistic Inference",
         authors="Wei Li; Pascal Poupart; Peter van Beek",
         doi="10.1613/jair.3232", published_pages="729–765 (37 pages)",
         version="Author-deposited arXiv PDF carrying the published JAIR layout and pagination",
         file="papers/JAIR/2011_Li_Exploiting_Structure_WMC.pdf",
         download_url="https://arxiv.org/pdf/1401.3886",
         publication_record="https://arxiv.org/abs/1401.3886"),
    dict(id="J2", journal="JAIR", year=2018,
         title="An Exhaustive DPLL Algorithm for Model Counting",
         authors="Umut Oztok; Adnan Darwiche", doi="10.1613/jair.1.11201",
         published_pages="1–32 (32 pages)", version="Publisher PDF",
         file="papers/JAIR/2018_Oztok_Exhaustive_DPLL.pdf",
         download_url="https://jair.org/index.php/jair/article/download/11201/26412",
         publication_record="https://jair.org/index.php/jair/article/view/11201"),
    dict(id="J3", journal="JAIR", year=2025,
         title="Certified Knowledge Compilation with Application to Formally Verified Model Counting",
         authors="Randal E. Bryant; Wojciech Nawrocki; Jeremy Avigad; Marijn J. H. Heule",
         doi="10.1613/jair.1.15958", published_pages="2057–2099 (43 pages)",
         version="Publisher PDF", file="papers/JAIR/2025_Bryant_Certified_Knowledge_Compilation.pdf",
         download_url="https://jair.org/index.php/jair/article/download/15958/27166",
         publication_record="https://jair.org/index.php/jair/article/view/15958"),
    dict(id="A1", journal="Artificial Intelligence", year=2008,
         title="On Probabilistic Inference by Weighted Model Counting",
         authors="Mark Chavira; Adnan Darwiche", doi="10.1016/j.artint.2007.11.002",
         published_pages="772–799 (28 pages)",
         version="Publicly hosted author manuscript, dated 4 November 2007; 48 physical pages, not publisher pagination",
         file="papers/AIJ/2008_Chavira_Probabilistic_Inference_WMC_author.pdf",
         download_url="https://www.cs.ru.nl/~peterl/BN/InferChavira2008.pdf",
         publication_record="https://www.sciencedirect.com/science/article/pii/S0004370207001889"),
    dict(id="A2", journal="Artificial Intelligence", year=2022,
         title="Approximate weighted model integration on DNF structures",
         authors="Ralph Abboud; İsmail İlkan Ceylan; Radoslav Dimitrov",
         doi="10.1016/j.artint.2022.103753", published_pages="Article 103753 (25 pages)",
         version="Publisher PDF deposited in Oxford Research Archive",
         file="papers/AIJ/2022_Abboud_Approximate_WMI_DNF.pdf",
         download_url="https://ora.ox.ac.uk/objects/uuid%3A3b540327-c8e1-4c1b-92cc-1ab139bf2fea/files/ssq87bw376",
         publication_record="https://ora.ox.ac.uk/objects/uuid%3A3b540327-c8e1-4c1b-92cc-1ab139bf2fea"),
    dict(id="A3", journal="Artificial Intelligence", year=2024,
         title="Enhancing SMT-based Weighted Model Integration by Structure Awareness",
         authors="Giuseppe Spallitta; Gabriele Masina; Paolo Morettin; Andrea Passerini; Roberto Sebastiani",
         doi="10.1016/j.artint.2024.104067",
         published_pages="Article 104067; publisher physical page count not verified",
         version="arXiv v2 author manuscript, 9 January 2024; 60 physical pages including appendices",
         file="papers/AIJ/2024_Spallitta_Structure_Aware_WMI_arxiv.pdf",
         download_url="https://arxiv.org/pdf/2302.06188",
         publication_record="https://arxiv.org/abs/2302.06188"),
    dict(id="A4", journal="Artificial Intelligence", year=2025,
         title="Lifted Inference beyond First-Order Logic",
         authors="Sagar Malhotra; Davide Bizzaro; Luciano Serafini",
         doi="10.1016/j.artint.2025.104310", published_pages="Article 104310 (29 pages)",
         version="arXiv v5 author manuscript, 25 February 2025; 55 physical pages including highlights and appendices",
         file="papers/AIJ/2025_Malhotra_Lifted_Inference_arxiv.pdf",
         download_url="https://arxiv.org/pdf/2308.11738",
         publication_record="https://repositum.tuwien.at/handle/20.500.12708/212807"),
    dict(id="I1", journal="International Journal of Approximate Reasoning", year=2017,
         title="Weighted positive binary decision diagrams for exact probabilistic inference",
         authors="Giso H. Dal; Peter J. F. Lucas", doi="10.1016/j.ijar.2017.08.003",
         published_pages="411–432 (22 pages)", version="Publisher-layout PDF on author university site",
         file="papers/IJAR/2017_Dal_Weighted_Positive_BDD.pdf",
         download_url="https://www.cs.ru.nl/P.Lucas/wpbdd.pdf",
         publication_record="https://www.sciencedirect.com/science/article/pii/S0888613X17305066"),
    dict(id="I2", journal="International Journal of Approximate Reasoning", year=2021,
         title="A compositional approach to probabilistic knowledge compilation",
         authors="Giso H. Dal; Alfons W. Laarman; Arjen Hommersom; Peter J. F. Lucas",
         doi="10.1016/j.ijar.2021.07.007", published_pages="38–66 (29 pages)",
         version="Publisher PDF deposited in University of Twente repository",
         file="papers/IJAR/2021_Dal_Compositional_Knowledge_Compilation.pdf",
         download_url="https://ris.utwente.nl/ws/portalfiles/portal/300192046/1_s2.0_S0888613X21001122_main.pdf",
         publication_record="https://research.utwente.nl/en/publications/a-compositional-approach-to-probabilistic-knowledge-compilation/"),
]

for record in RECORDS:
    path = ROOT / record["file"]
    raw = path.read_bytes()
    assert raw.startswith(b"%PDF-"), f"Not a PDF: {path}"
    reader = PdfReader(path, strict=True)
    record.update(
        downloaded_on="2026-10-14",
        bytes=len(raw),
        sha256=hashlib.sha256(raw).hexdigest(),
        physical_pdf_pages=len(reader.pages),
        all_pages_text_extractable=all(bool((page.extract_text() or "").strip()) for page in reader.pages),
    )
    assert record["all_pages_text_extractable"], path

(ROOT / "manifest.json").write_text(json.dumps(RECORDS, ensure_ascii=False, indent=2) + "\n")
print(f"Validated {len(RECORDS)} PDFs: " + ", ".join(f'{r["id"]}={r["physical_pdf_pages"]}p' for r in RECORDS))
