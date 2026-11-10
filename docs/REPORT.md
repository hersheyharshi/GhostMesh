# GhostMesh: Predictive, Trust-Aware Routing for Decentralised Emergency Mesh Networks

**Harshita Pandita** · Technical report · v1.0 · November 2026

---

## Abstract

Emergency mesh networks must carry SOS traffic over relays that fail as a disaster spreads and that may be compromised. We present **GhostMesh**, a hop-by-hop link-state routing protocol whose edge cost fuses four signals computed entirely on the nodes: (i) a Bayesian trust model with asymmetric, channel-aware evidence and credibility-weighted recommendations; (ii) verification of advertised link quality against first-hand measurement; (iii) a node-failure forecaster trained by *gossip learning*, a server-free form of federated learning; and (iv) stigmergic pheromone reinforcement. In a discrete-time simulation of 50 nodes under a spreading hazard front with 20% insider attackers (black-hole, grey-hole, on-off, all bad-mouthing and advertising forged links), GhostMesh delivers **58.0%** of SOS packets versus **47.2%** for OLSR-ETX and **53.3%** for a Beta-reputation router (paired over 20 scenarios; *p* < 10⁻³ for both). It reduces packets lost to attackers by 66% and matches epidemic flooding's delivery at 45% less energy per delivered packet. Its reputation separates attackers from honest nodes with AUC 0.88, where the trust baseline is inverted by bad-mouthing (AUC 0.16). Ablations show link-claim verification is the single most valuable component. Two components, the failure forecaster and pheromone, have no significant effect on delivery; we analyse why. As a learning problem, the gossip-learned forecaster (AUC 0.817) outperforms isolated local learning (0.801) and matches a centralised offline model (0.800).

---

## 1. Problem

A disaster removes cellular infrastructure. Survivors' phones, rescuers' radios, drones and sensors form a multi-hop wireless mesh that must forward short SOS messages to a few **gateways** with satellite or drone uplinks. Two properties of this setting break the assumptions of standard MANET routing:

* **Correlated, progressive failure.** Nodes die from the hazard (fire, flood) or from battery exhaustion, which is accelerated by relaying. A dead node is not noticed until its neighbours' HELLO timers expire, so for several steps it remains in routing tables as a **ghost**. A packet held by a node when it dies is lost.
* **Insiders.** Nodes can be compromised. Because relays are anonymous members of the public, admission control is impossible and misbehaviour must be inferred from behaviour.

## 2. System and threat model

**Network.** *N* = 50 static nodes uniformly placed in a unit square, of which 3 are gateways at the edges. Nodes within range *R* = 0.24 share a link with delivery ratio

  pdr(i,j) = σ(−(d_ij/R − 0.72)/0.075 + f_ij(t)) · (1 − 0.45·max(s_i, s_j)),

where f is symmetric AR(1) shadow fading (stationary σ = 0.7 logits) and s is environmental stress near the hazard. Time is discrete (300 steps). Each node may transmit 3 packets per step; a packet is retried up to 3 times before the link is blacklisted for 15 steps. Packets live 40 steps and are buffered when no route exists (store-carry-forward).

**Traffic.** Honest non-gateway nodes generate Poisson(1.5) SOS packets per step, anycast to any gateway.

**Energy.** Battery ~ U(0.45, 1.0); idle drain 7×10⁻⁴/step (beacons included), 4×10⁻³ per transmission, 10⁻³ per reception, 3×10⁻⁴ per control-flood transmission.

**Hazard.** From step 30 a circular front expands at 0.0012/step from an epicentre at least 0.35 from every gateway. Inside it, death probability is 0.035·(1 + 10·min(depth, 0.15)) per step; a 4×10⁻⁴ hardware failure rate applies everywhere. Dead nodes are detected after 6 steps.

**Adversary.** 20% of relays, chosen uniformly at random, become malicious at step 20, cycling through three strategies: **black-hole** (drop everything), **grey-hole** (drop with probability 0.5), **on-off** (black-hole for 20 steps, honest for 20). While active, every attacker (a) **forges link-state advertisements**, claiming ETX = 1 on all its links, and in AODV answers route requests claiming a one-hop route to a gateway; (b) **bad-mouths**, recommending honest nodes as fully untrustworthy and colluders as fully trustworthy; (c) advertises low failure risk; and (d) mimics HOLD notices to delay detection. Attackers do not generate traffic.

## 3. GhostMesh

Every node *u* holding a packet computes a least-cost path from itself to a virtual sink connected to all gateways (Dijkstra; nodes the packet has already visited are excluded to prevent loops) and forwards to the first hop. The cost of edge *x → y* **as seen by u** is

  c_u(x,y) = ETX(x,y) · T_u(y)^−γ · (1 + κ · r_y · (1 + q_y / C)) / (1 + ρ · τ_xy),   with T_u(y) < θ ⇒ c = ∞,

with γ = 2, κ = 4, ρ = 0.6, θ = 0.35, queue length q_y and capacity C. Each factor is described below.

### 3.1 Trust (T)

Node *u* keeps Beta(α_uk, β_uk) about every node *k*, with prior Beta(2, 1).

*Watchdog.* After handing a packet to *k*, *u* listens for *k*'s retransmission and overhears it with probability min(1, pdr(u,k) + 0.15). Overheard ⇒ α += 1. If the 4-step deadline passes, a node still holding the packet may issue a HOLD notice (attackers mimic this); after at most two extensions, β += w_neg · q, where:

* **Asymmetric evidence.** w_neg = 1.5, and evidence above the prior decays with λ⁺ = 0.985 for good and λ⁻ = 0.998 for bad evidence. An on-off attacker cannot launder its reputation during its honest phase, and a grey-hole's partial forwarding does not offset its drops.
* **Channel-aware evidence.** q = min(1, pdr(u,k) + 0.15) is the probability that a forward *would* have been overheard. Silence on a bad link is weak evidence.

*Recommendations.* Every 5 steps neighbours exchange first-hand evidence, with any single recommendation capped at 5 units. Each node keeps a second Beta, the **credibility** C_uj of recommender *j*: whenever *j* recommends about a node *u* has ≥ 2 units of first-hand evidence on, agreement within 0.3 increments credibility and disagreement decrements it (decay 0.97). Only recommenders with credibility ≥ 0.5 and direct trust ≥ 0.5 are heard, weighted by C², and a recommendation that deviates by > 0.4 from *u*'s own opinion (when *u* has ≥ 3 units of evidence) is dropped. Indirect evidence enters with discount 0.35 and is recomputed each round rather than accumulated, avoiding echo-chamber feedback. This separation of *functional* trust (does *j* forward?) from *recommendation* trust (does *j* tell the truth?) is what defeats bad-mouthing.

### 3.2 Link-claim verification

Each step every honest node compares the ETX that each neighbour *v* advertises for link (v, u) with its own noisy measurement of the same link (σ = 0.04 on delivery ratio). If measured ETX > 1.5 × claimed ETX on a link with measured delivery > 0.2, then β_uv += 0.3. Inflated claims, the mechanism behind route attraction, therefore generate evidence without a single packet being lost.

### 3.3 Ghost forecaster (r)

Every node computes a feature vector from its HELLO telemetry: battery, **battery urgency** exp(−TTE/H) where TTE is time-to-empty at the current drain rate, stress sensor reading (σ = 0.12 noise) and its slope, fraction of neighbours lost in the last 10 steps, and mean link quality. A logistic model predicts *P(dies within H = 12 steps)*.

*Learning without a server.* A snapshot taken at *t* is labelled at *t + H*. A dead node cannot train on its own death, so every node that heard its beacons (its neighbours at time *t*) performs the SGD step instead (lr 1.5, positive-class weight 2, L2 10⁻⁴). After each step every node replaces its weights with the average over itself and its live neighbours (row-stochastic gossip mixing). Each node advertises its own predicted risk r. Attackers advertise 0.02.

*Ghost evacuation.* A node with r ≥ 0.3 and no route transfers custody of buffered packets to its lowest-risk trusted neighbour, provided that neighbour's risk is less than half its own.

### 3.4 Pheromone (τ)

On delivery, every edge on the packet's path gains 1/(path length) pheromone; τ evaporates by 5% per step; watchdog failures halve all pheromone towards the offender and zero the observer's own link to it.

## 4. Baselines

| Baseline | Model |
|---|---|
| **Flooding** | Every node rebroadcasts each packet once (duplicate suppression); flood stops at first gateway delivery. Attackers absorb copies. |
| **AODV** | Reactive min-hop discovery, charged as a network-wide RREQ flood, route caching, route-error invalidation, local repair. Attackers win route replies with forged one-hop claims (the classic AODV black-hole). |
| **OLSR-ETX** | Proactive link-state, hop-by-hop least-ETX anycast routing on advertised (forgeable) link qualities. |
| **Trust-ETX** | OLSR-ETX cost divided by T^γ with the same γ and exclusion threshold as GhostMesh; standard Beta reputation with symmetric forgetting and trust-weighted, unfiltered recommendations. |

All routers share the same world model, watchdog-free MAC, blacklisting after failed retries, store-carry-forward buffer and TTL. Each scenario seed fixes the topology, attackers, fading, hazard and failure randomness through separate random streams, so every protocol faces the **identical** disaster and comparisons are paired.

## 5. Evaluation

20 seeds for the main benchmark and ablation, 10 per point for the attack sweep and forecaster. We report means with 95% percentile-bootstrap CIs (4,000 resamples) and two-sided Wilcoxon signed-rank tests on per-seed paired differences. The full study takes about 6 minutes on 2 CPU cores (`experiments/run_experiments.py`). Raw results are in `results/`.

### 5.1 Main benchmark (20% attackers)

| | Flooding | AODV | OLSR-ETX | Trust-ETX | **GhostMesh** |
|---|---|---|---|---|---|
| Delivery ratio | 60.3% | 40.2% | 47.2% | 53.3% | **58.0%** |
| 95% CI | [56.2, 63.9] | [36.4, 44.1] | [42.6, 51.7] | [47.9, 58.3] | [52.1, 63.6] |
| Lost to attackers | n/a | 52.3% | 42.5% | 34.1% | **14.6%** |
| Lost with dying node | 0 | 4.7% | 3.0% | 3.6% | 6.2% |
| Mean latency (steps) | 2.8 | 4.1 | 4.2 | 4.4 | 5.3 |
| Energy / delivery (×10⁻³) | 75.0 | 81.6 | 41.5 | 38.1 | 40.9 |
| Relays alive at t = 300 | 30.3% | 42.7% | 54.6% | 54.4% | 50.3% |
| Detection AUC | n/a | n/a | n/a | 0.16 | **0.88** |

Paired delivery difference, GhostMesh minus baseline:

| vs. | Δ (pts) | 95% CI | GhostMesh better | Wilcoxon *p* |
|---|---|---|---|---|
| AODV | +17.8 | [14.2, 21.5] | 20 / 20 | 1.9×10⁻⁶ |
| OLSR-ETX | +10.8 | [8.0, 14.0] | 20 / 20 | 1.9×10⁻⁶ |
| Trust-ETX | +4.7 | [2.6, 6.9] | 16 / 20 | 8.5×10⁻⁴ |
| Flooding | −2.3 | [−5.3, +0.9] | 6 / 20 | 0.22 (n.s.) |

The gain over OLSR-ETX ranges from +1.5 pts (seed 12) to +30.3 pts (seed 13), median +8.8 (seed 4). The dashboard ships these three scenarios.

### 5.2 Attack intensity

| Attackers | 0% | 10% | 20% | 30% | 40% |
|---|---|---|---|---|---|
| Flooding | 63.6 | 60.9 | 59.4 | **55.2** | **49.0** |
| AODV | 65.0 | 47.2 | 38.0 | 28.3 | 27.1 |
| OLSR-ETX | **70.0** | 57.2 | 47.0 | 37.7 | 33.3 |
| Trust-ETX | 69.3 | 58.5 | 53.9 | 42.8 | 37.9 |
| GhostMesh | 69.7 | **63.8** | 58.1 | 48.3 | 43.8 |

GhostMesh carries no measurable peacetime cost and leads all unicast protocols under attack. Flooding is the most robust to very high attacker densities. It pays for that at roughly 1.6–2× the energy per delivery and loses nodes much faster (30% alive at the end vs. 50%).

### 5.3 Ablation (paired, 20 seeds)

| Removed component | Δ delivery (pts) | Δ lost to attackers (pts) | Δ detection AUC |
|---|---|---|---|
| Link-claim verification | **−3.0** [−4.2, −1.7] | **+9.3** [7.8, 10.9] | **−0.27** [−0.35, −0.21] |
| Asymmetric evidence | −0.5 [−2.1, +1.1] | **+7.7** [5.7, 9.8] | **−0.08** [−0.15, −0.02] |
| Recommendation credibility | −0.3 [−2.3, +1.4] | **+1.8** [0.7, 3.0] | **−0.11** [−0.14, −0.08] |
| Ghost forecaster | −0.3 [−1.9, +1.3] | 0.0 [−1.3, +1.5] | −0.02 [−0.05, +0.01] |
| Pheromone | +1.2 [−0.6, +3.2] | −0.5 [−1.6, +0.6] | −0.02 [−0.04, 0.00] |
| All trust machinery | **−5.9** [−7.8, −3.9] | **+23.0** [20.2, 25.8] | n/a |

### 5.4 Ghost forecaster

Evaluated prequentially on the later half of each run (predictions made before labels exist): 58,572 node-steps, 1,877 deaths within the horizon.

| Model | AUC, mean of 10 seeds [95% CI] |
|---|---|
| Gossip-learned (GhostMesh) | **0.817** [0.781, 0.850] |
| Local-only learning | 0.801 [0.778, 0.825] |
| Centralised logistic regression (offline, trained on the first half) | 0.800 [0.676, 0.879] |
| Battery level only | 0.658 [0.601, 0.707] |
| Stress sensor only | 0.843 [0.815, 0.871] |

## 6. Discussion

### 6.1 Where the gains come from
Almost all of GhostMesh's advantage is security: losses to attackers fall from 42.5% to 14.6%. The ablation attributes the largest share to **link-claim verification**. Route attraction relies on a forged advertisement that the victim's own radio can contradict, so the lie becomes evidence before any packet is lost. Asymmetric evidence and recommendation credibility each substantially reduce attacker throughput and improve detection, but in isolation their delivery effect sits within noise, because the remaining defences catch most of what they would have caught. Trust-ETX shows the cost of naive recommendations: bad-mouthing inverts its reputation ordering (AUC 0.16), so its improvement over OLSR-ETX comes from direct observation alone.

### 6.2 Why the forecaster does not move delivery
The forecaster predicts well (AUC 0.82) yet has no significant effect on delivery. Two mechanisms cancel: penalising high-risk relays reduces packets lost with dying carriers, but in a disaster the high-risk region is often where the packets *are*, so avoiding it lengthens paths and buffering time and more packets expire. In a development sweep (no attackers, faster front, an earlier version of the forecaster features), raising the penalty κ from 0 to 15 cut losses with dying nodes from 12.1% to 9.3% of packets while expired packets rose by a similar amount, leaving delivery unchanged. Ghost transmissions (attempts to dead-but-advertised nodes) were rare in our model because three failed retries blacklist a link, which limits what forecasting can save. We expect the forecaster to matter more with mobility, longer HELLO timeouts, or delay-tolerant traffic with long TTLs. We report this as a negative result rather than tuning the scenario until it disappears.

### 6.3 Decentralised learning
Gossip averaging improves on isolated learning (0.817 vs. 0.801) and matches a centralised model given everyone's data offline (0.800), whose wide CI reflects distribution shift as the front grows. A single heat sensor remains a strong baseline (0.843) because hazard deaths dominate. The learned model trades a little of that for coverage of battery deaths, which the sensor cannot see.

### 6.4 Costs
GhostMesh is ~1 step slower on average (5.3 vs. 4.2) because it detours around distrusted relays and buffers rather than forwarding into them, and it ends with fewer relays alive (50% vs. 55% for OLSR-ETX), most likely because routing around distrusted nodes concentrates relaying on fewer honest ones. Its false-positive rate (8% of surviving honest nodes below 0.4 reputation) comes mostly from honest dead-end relays whose only onward path runs through attackers. They hold packets they cannot forward, which looks like dropping.

## 7. Limitations

* **Radio/MAC abstraction.** No collisions, hidden terminals or interference; capacity is a per-node transmission budget. Control traffic for link-state protocols is folded into idle energy, while AODV floods are charged explicitly.
* **Static nodes.** Rescuers move; mobility would change both link dynamics and the value of forecasting.
* **Non-adaptive attackers.** Attackers follow fixed strategies. An attacker that understands the defences (e.g. lies only on near-perfect links, or keeps its drop rate just under the detection rate) is future work.
* **Watchdog idealisation.** Overhearing is modelled probabilistically; directional antennas, power control or collusive relaying could defeat it.
* **Scale.** 50 nodes, 300 steps. Per-node Dijkstra and O(N³) recommendation filtering are fine at this scale but need sparsification beyond a few hundred nodes.
* **Single scenario family.** One hazard geometry, one traffic pattern. Results should be validated in ns-3/OMNeT++ and on a hardware testbed (e.g. LoRa or ESP-NOW mesh).

## 8. Future work

Adversarially adaptive attackers trained with reinforcement learning against GhostMesh; mobility models for rescue teams; replacing the logistic forecaster with a small graph neural network trained by gossip; secure aggregation for the gossip step so that poisoned model updates from attackers are bounded; and porting the router to an ESP32/LoRa testbed.

## References

1. C. Perkins, E. Belding-Royer, S. Das. *Ad hoc On-Demand Distance Vector (AODV) Routing.* RFC 3561, 2003.
2. T. Clausen, P. Jacquet. *Optimized Link State Routing Protocol (OLSR).* RFC 3626, 2003.
3. D. De Couto, D. Aguayo, J. Bicket, R. Morris. *A High-Throughput Path Metric for Multi-Hop Wireless Routing.* MobiCom 2003.
4. S. Marti, T. Giuli, K. Lai, M. Baker. *Mitigating Routing Misbehavior in Mobile Ad Hoc Networks.* MobiCom 2000.
5. S. Buchegger, J.-Y. Le Boudec. *Performance Analysis of the CONFIDANT Protocol.* MobiHoc 2002.
6. A. Jøsang, R. Ismail. *The Beta Reputation System.* Bled eConference 2002.
7. G. Di Caro, M. Dorigo. *AntNet: Distributed Stigmergetic Control for Communications Networks.* JAIR 9, 1998.
8. R. Ormándi, I. Hegedűs, M. Jelasity. *Gossip Learning with Linear Models on Fully Distributed Data.* Concurrency and Computation 25(4), 2013.
9. B. McMahan et al. *Communication-Efficient Learning of Deep Networks from Decentralized Data.* AISTATS 2017.
10. Y. Sun, Z. Han, W. Yu, K. J. R. Liu. *A Trust Evaluation Framework in Distributed Networks: Vulnerability Analysis and Defense Against Attacks.* INFOCOM 2006.
