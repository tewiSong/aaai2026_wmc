Paper Decision
Decisionby Program Chairs24 Sept 2026, 21:46 (modified: 25 Sept 2026, 13:33)Program Chairs, Authors, Program Committee Submitted, Area Chairs, Secondary Senior Program Committee, Senior Program CommitteeRevisions
Decision: Reject
Comment:
This paper did not advance to Phase 2 review for the AAAI-27 conference. The paper was reviewed by at least two human reviewers and at least one senior member of the program committee in Phase 1. In addition, papers with at least one positive review were examined by one program chair or associate program chair to ensure the rejection was justified. This decision is final. The provided reviews contain more detail, which we hope will be beneficial to the progress of this research.

Experiments do not test the method where it is needed
Official Reviewby Program Committee c36U28 Aug 2026, 14:56 (modified: 06 Oct 2026, 22:28)Program Chairs, Area Chairs, Senior Program Committee, Program Committee c36U, Authors, Program Committee SubmittedRevisions
Paper Summary:
Neural predictors assign independent scores to ontology terms, and those scores violate the ontology's own logic by ranking a specialized term above the general term containing it. The paper treats predictor scores as Bernoulli parameters and conditions the resulting product distribution on the ontology, keeping only valid labelings and renormalizing. The normalizer of that conditional distribution is a weighted model count, which motivates the focus on WMC.

Exact WMC scales exponentially in the structural width of the constraint graph, and the paper reports that monolithic junction tree exhausts a 300 GB budget on four of five benchmarks. Cheap local message passing is the natural alternative, but the paper shows with a four-variable example that it fails on reconvergent structure: when a term has two parents sharing an ancestor, independent messages double-count the shared evidence.

The proposed method selects a subset of variables as a core. The remaining variables fall into modules, defined as the connected components left after removing the core. Each module is eliminated exactly, summing out its interior while holding fixed the core variables it touches, called its boundary. The result is a separator factor: one joint function over the boundary that preserves the correlations the module induces. Inference then runs over the reduced core, and eliminated marginals are recovered by back substitution. The construction recurses when the reduced core remains too wide.

Separator factors are exponential in boundary size, which is infeasible for the boundaries reported here, up to 1,082 variables. For those, the factor is queried pointwise and fitted with a nonnegative low-rank tensor train, with logical support handled separately so the fit cannot place mass on unsatisfiable interfaces.

Two results are stated: exact separators reproduce monolithic WMC, and module marginal error is bounded by boundary posterior error. Experiments report machine precision agreement with exact references on GO cellular component and 432 subgraphs, all-term recovery on five ontologies up to 205,592 terms, a rank ablation, and a CAFA protein function study.

Strengths:
The paper tackles an important problem: predictor scores that violate the hierarchy they were trained against, corrected at inference time without retraining.
The paper is explicit that reading scores as Bernoulli parameters is a modeling choice and that guarantees concern conditioning rather than calibration.
The diamond example makes the case for joint boundary summarization with arithmetic that a reader can verify by hand.
The decomposition is well defined. Deriving modules from the core makes the key property, that no constraint straddles two modules, automatic rather than assumed.
The exact mode validation is convincing on its own terms: machine precision agreement across hundreds of instances.
Weaknesses:
The three headline claims cover three disjoint regimes. Exact recovery, completion on complete vocabularies, and error reduction with rank read as properties of one system but are properties of three settings that never overlap. Exactness appears only where a monolithic reference exists. The rank study covers small enumerable modules. On the four large ontologies, the reported quantity is completion, meaning that marginals were produced. No accuracy claim about those marginals appears anywhere. Any empirical handle on accuracy at scale would help, including a self-consistency check such as agreement between successive ranks.

The benchmark carrying exactness and the entire downstream study is the one where the method does not engage. Table 2 reports GO cc with 100% coverage, one module, depth zero, and maximum boundary size zero, indicating an empty core and no decomposition. The timing agrees: modular WMC is roughly six times slower than monolithic JT on that row. GO cc also carries Table 1 and every number in Table 4. So the exactness check largely verifies correct degeneration to the baseline, and the only application-level evaluation runs in the inactive configuration. A CAFA-style evaluation on GO bp or GO mf, where the decomposition is required, would address this.

The diamond argues that boundary factors must not be factorized across boundary variables. At scale the method represents them as low-rank tensor trains, a structured factorization of that dependence. This may be fine, since low rank is much weaker than independence, but no argument is given for why ontology-induced separators should be compressible at rank 8 across a thousand-variable boundary. The ablation establishes that rank 1 is inadequate, which the diamond already showed, but not that rank 8 is adequate at that width.

The scale comparison is against monolithic junction tree, reported as out of memory. The paper cites ADDMC and DPMC without running them, and loopy BP is used downstream but not at scale despite producing all-term marginals on these vocabularies. Even timeout or out-of-memory results under the same budget would strengthen the comparison.

Justification of Recommendation:
Mostly because of weaknesses 1 and 2.

The exact mode is sound, and several design choices are good, including deriving modules from the core, separating logical support from the fit, and the direct product control. However, exactness validation is carried by a benchmark whose reported structure indicates the decomposition does not engage (weakness 2), and that same benchmark carries the entire downstream evaluation, so the scaling contribution is never shown to affect any application-level quantity. Where the method does engage, only completion is reported (weakness 1).

A higher rating would have followed from a downstream evaluation on a namespace where the decomposition is required, or from any empirical evidence that the marginals recovered at scale are accurate. Should the rebuttal establish that the exactness validation exercises the decomposition rather than reducing to monolithic inference, this assessment would move upward.

Specific Points of Feedback for Rebuttal:
Table 2 shows GO cc with 100% coverage, one module, depth 0, and maximum boundary size 0. Does the selector return an empty core there, so the run reduces to monolithic junction tree? If so, does Table 1 exercise module elimination and recovery at all?
How were the 432 reference subgraphs selected, and what are their ranges of size, width, and boundary size?
Which rows of Table 2 used exact separators and which used tensor train separators, and at what rank?
Is any evidence available that recovered marginals at scale are accurate, for example, agreement between rank 4 and rank 8 on the same instance, or exact recovery on terms whose modules remain enumerable?
Is there any comparison against ADDMC, DPMC, or d4 on the large ontologies?
Rating: 4: Ok but not good enough - rejection
Confidence: 2: The reviewer is willing to defend the evaluation, but it is quite likely that the reviewer did not understand central parts of the paper
A new weighted model counting algorithm for ontology-based instances
Official Reviewby Program Committee KEL125 Aug 2026, 08:50 (modified: 06 Oct 2026, 22:28)Program Chairs, Area Chairs, Senior Program Committee, Program Committee KEL1, Authors, Program Committee SubmittedRevisions
Paper Summary:
The paper introduces a weighted model counting (WMC) algorithm designed to be efficient on ontology-based probabilistic inference queries. Ontologies provide a unique challenge to WMC algorithms in that while the logical structure of the propositional formula is relatively simple, the number of variables can be orders of magnitude higher than in other domains. The key idea is to partition the constraint graph into one core and one or more modules. The variables within each module can then be partitioned into boundary variables that have neighbors in the core and internal variables. Internal variables can be eliminated, preserving their (exact or approximate) effect on the boundary variables as a factor, i.e., a function that returns a value for each possible combination of boundary variable values. The experimental results show promise, particularly in significantly lower memory usage.

Strengths:
The approach seems to be correct and at least moderately novel.

Weaknesses:
The paper reads as if it starts on page 3, when the topic has already been introduced and all the relevant terms have been defined. So the reader is left wondering: 1) what problem is being solved here, i.e., what is the exact setup? 2) what do all these terms mean? Which ones are standard, and which ones are introduced by the paper?
Similarly, the paper unnecessarily mixes terminology from many areas (WMC, graphical models, tensor-based approximations, neurosymbolic artificial intelligence (NeSy), and a few others), creating a unique perspective on the subject and assuming that the reader is already familiar with that perspective.
The novelty should be made more explicit by comparing it to related methods such as tree-decomposition-based WMC [Dudek, Phan, & Vardi, 2020], tensor-network-based WMC [Dudek & Vardi, 2020], and classical algorithms for probabilistic inference in graphical models.
There is no way to tell whether the proposed algorithm is significant because it is not being compared to the state of the art in exact and heuristic/approximate WMC. See the results of the model counting competition (https://mccompetition.org/) from this year and previous iterations for suggestions on what solvers should be included. Even if these algorithms run out of memory while reading the input file, that should still be explicitly mentioned in the paper.
Justification of Recommendation:
The paper needs to be almost entirely rewritten, and the experiments should include state-of-the-art WMC algorithms. Since the proposed algorithm appears to be largely based on well-established ideas such as the min-fill heuristic, a deeper explanation of what makes it succeed where other algorithms fail (if that is the case) should be provided. Moreover, the paper has essentially zero examples.

Currently, positioning this paper in the context of machine learning and NeSy seems like a red herring: it obfuscates the key problem addressed in the paper and requires more contextual knowledge from the reader. While listing such areas of application is valuable, if there is a reason not to interpret this paper as proposing a generic WMC algorithm tailored for large-but-easy instances, that reason should be clearly specified.

Specific Points of Feedback for Rebuttal:
Even the abstract jumps into details way too quickly. What problem is being solved? What problem is WMC being applied to?
Similarly, the introduction should be more high-level, leaving the technical setup for later sections.
Define terms such as "monolithic WMC" and "directed path-wise update".
Be more precise about "(structural) width": does this term always refer to treewidth and similar complexity measures? Spell it out on a case-by-case basis.
What is an endpoint? What does it mean to "clip" an endpoint score?
Define Horn clauses. Also, why specify that they are "grounded"?
Clarify the connection between this work and factor graphs and variable elimination.
It would be helpful to mention that, in this context, "constraint graph" is also known as "primal graph".
 and 
 were never defined (and are only used once).
The paper would benefit from more signposting: "in this section", "in the next section", "paper organization", etc.
Rating: 3: Clear rejection
Confidence: 4: The reviewer is confident but not absolutely certain that the evaluation is correct
AI Review
AI Reviewby AI Reviewer ()27 Jul 2026, 14:59 (modified: 25 Sept 2026, 00:43)Program Chairs, AI Reviewer, Area Chairs, Senior Program Committee, Authors, Program Committee SubmittedRevisions
Review:
Title: Scaling Ontology-Conditioned Weighted Model Counting with Modular Separators

Synopsis of the paper
The paper decomposes ontology-conditioned weighted model counting into peripheral modules connected through joint boundary factors. Exact factors preserve the monolithic posterior, while nonnegative tensor-train approximations handle wide boundaries without dense tables. Recursive core contraction and backward calibration are intended to return marginals for every ontology term. Experiments examine exact preservation, resource scaling on biomedical ontologies, separator approximation, and fixed-score protein-function prediction.

Summary of Review
The paper addresses the width bottleneck in ontology-conditioned WMC through a coherent combination of modular elimination and low-rank boundary representations. Proposition 1 and Table 1 substantiate the exact mode, while Table 2 demonstrates completion beyond the resource limits of the matched monolithic junction-tree implementation. The central technical concern is that recovery of eliminated-variable marginals across wide approximate boundaries is not operationally specified: the displayed recovery formula remains exponential in boundary size. Moreover, the production experiments establish completion and held-out separator fit, but not end-to-end accuracy of the resulting ontology marginals. The novelty boundary and comparative scalability also require sharper treatment relative to compositional and tensor-based model counting.

Strengths

Joint separator factors directly address the dependence lost by path-wise summaries. The diamond construction in “Motivation: Reconvergent Dependence” and the factor partition preceding Proposition 1 explain why a multi-variable boundary must be retained jointly, aligning the decomposition with the globally normalized WMC target.

The exact-mode evaluation tests the central preservation property against genuine references. Table 1 compares complete recovered marginal sets with monolithic inference on full GO cellular component and 432 retained subgraphs, reaching numerical-precision errors; this validates factor assignment, separator elimination, and reverse recovery together rather than testing only the normalizer.

The approximation study distinguishes representational capacity from query-fitted behavior. Figure 3(b) uses an enumerable exact separator, while Table 3 evaluates recovered marginals on reference modules and held-out queries on production boundaries; the consistent rank-dependent error reduction provides evidence that joint low-rank structure is useful.

The experimental design separates exactness, approximation behavior, resource scaling, and downstream correction across RQ1–RQ3. Table 2 also uses the same junction-tree backend monolithically and within modules, reducing backend differences as a confound in the resource comparison.

Weaknesses

Scalable all-term recovery is not operationally specified. The recovery equation sums 
 over all 
 boundary assignments, while Table 2 reports 
 as large as 1,082; the statement that a calibrated module junction tree reuses its messages does not provide a compact representation of 
 or a contraction with an approximate TT boundary posterior. Because this computation produces the paper’s central all-term outputs, the main text needs a concise factorized recovery formula and complexity account.

Production-scale marginal accuracy is not established. Table 2 reports only completion, time, and memory, whereas the production rows of Table 3 report held-out separator NMAE rather than boundary-posterior or recovered-marginal error; exact marginal validation is confined to boundaries of size at most 12. NMAE, its query distribution, and the ranks and query budgets used in Table 2 are also unspecified, preventing the production separator fits from being connected quantitatively to the reported complete-ontology outputs.

The novelty boundary is insufficiently established against compositional and tensor-based model counting. Related Work cites Dal et al. (2021) and says that the framework “specializes these ideas,” but does not identify which prior compositional capabilities are absent; the reference list also omits recent tensor-counting and TT knowledge-compilation work (Goral et al., 2024; Onaka et al., 2025). This makes it difficult to isolate the contribution among core selection, recursive recovery, support-aware fitting, and TT-native contraction.

The scale comparison is too narrow to establish comparative scalability. Table 2 includes only the matched monolithic junction-tree backend, although the paper itself cites compositional WMC, DPMC, ADDMC, and knowledge-compilation approaches as relevant alternatives. The experiment isolates benefits relative to this backend, but does not establish how the implementation compares with contemporary exact or approximate model-counting systems.

Table 4 provides little evidence of a practically distinguishable advantage over loopy BP. Exact WMC exceeds BP by only 
 and 
 in Fmax, and by 
 and 
 in Brier score for DeepGOCNN and MLP, respectively; BP already gives zero or 
 violations, and no uncertainty intervals over the evaluation set are reported. Moreover, this experiment uses exact junction-tree WMC rather than the scalable TT mode.

Suggestions for Improvement

A rank-aware decomposition objective could connect structural selection more directly to approximation cost. Algorithm 1 optimizes a boundary-last width cap using a static degree-based ranking, whereas “Computational scope” shows that boundary size, oracle-query cost, TT rank, and bond-variable elimination jointly determine runtime.

A worked recursive example would clarify the interaction among existing separator factors, forced module merges, TT bond variables, and reverse calibration. The current “Recursive contraction” paragraph compresses these mechanisms into prose, making the multilevel factorization harder to reconstruct than the top-level construction in Figure 2.

An analysis relating required TT rank to boundary topology, reachability constraints, and score concentration could provide a more general account of when the approximation should work. The current rank results demonstrate empirical trends but do not connect compressibility to observable ontology structure.

Additional Details and Minor Issues

The update producing the path-wise value 
 in the diamond example should be stated in the main text.
Figure 3(b) uses TT-SVD on an exact table, whereas the production method uses nonnegative query-fitted ALS; this distinction should be emphasized.
Algorithm 1 does not state how inference proceeds after BUDGET-EXHAUSTED.
Approximate outputs should use notation such as 
 and 
 consistently rather than the exact 
 and 
.
“Complete ontology” should be qualified as the complete vocabulary with binary is-a implications, since other ontology axioms are not used.
References

Dal, G. H., Laarman, A. W., Hommersom, A., & Lucas, P. J. F. (2021). A compositional approach to probabilistic knowledge compilation. International Journal of Approximate Reasoning, 138, 38–66. (sciencedirect.com)

Goral, A., Giesen, J., Blacher, M., Staudt, C., & Klaus, J. (2024). Model counting and sampling via semiring extensions. Proceedings of the AAAI Conference on Artificial Intelligence, 38(18), 20395–20403. (ojs.aaai.org)

Onaka, R., Nakamura, K., Nishino, M., & Yasuda, N. (2025). Tensor decomposition meets knowledge compilation: A study comparing tensor trains with OBDDs. Proceedings of the AAAI Conference on Artificial Intelligence, 39(14), 15109–15117. (ojs.aaai.org)