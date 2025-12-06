# Blockchain-Powered HIPAA Redaction Implementation Guide

> **ONE-SENTENCE SUMMARY**: Blockchain lets you prove that redaction was done correctly — without ever revealing PHI — and makes that proof impossible to tamper with.

---

## Table of Contents

1. [Executive Summary](#executive-summary)
2. [Architecture Overview](#architecture-overview)
3. [Implementation Phases](#implementation-phases)
4. [Detailed Component Specifications](#detailed-component-specifications)
5. [Integration Points in Existing Codebase](#integration-points-in-existing-codebase)
6. [Cryptographic Design](#cryptographic-design)
7. [Smart Contract Specifications](#smart-contract-specifications)
8. [Zero-Knowledge Proof System](#zero-knowledge-proof-system)
9. [Testing Strategy](#testing-strategy)
10. [Deployment & Operations](#deployment--operations)
11. [Compliance Verification](#compliance-verification)
12. [Implementation Checklist](#implementation-checklist)

---

## Executive Summary

### What Blockchain Adds to HIPAA Redaction

| Capability | Traditional Logs | Blockchain-Powered |
|------------|-----------------|-------------------|
| Prove what PHI was removed | ❌ Trust-based | ✅ Cryptographic proof |
| Tamper-proof audit trail | ❌ Logs can be altered | ✅ Immutable ledger |
| Prove only PHI was removed | ❌ Cannot verify | ✅ Zero-knowledge proofs |
| Cross-institution trust | ❌ Requires trusted third party | ✅ Trustless verification |
| Legal defensibility | ⚠️ Weak | ✅ Mathematically provable |

### What Blockchain Does NOT Do

- ❌ Does NOT perform redaction (happens off-chain)
- ❌ Does NOT store PHI (only hashes and proofs)
- ❌ Does NOT validate clinical content
- ❌ Does NOT enforce HIPAA by itself

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           HIPAA REDACTION SYSTEM                             │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  ┌──────────────┐    ┌──────────────────┐    ┌──────────────────────────┐  │
│  │   INPUT      │───▶│  PHI DETECTION   │───▶│   REDACTION ENGINE       │  │
│  │   (Query/    │    │  LAYER           │    │   • Pattern masking      │  │
│  │   Response)  │    │  • NER Models    │    │   • Tokenization         │  │
│  └──────────────┘    │  • Regex Patterns│    │   • Rule application     │  │
│                      │  • ML Classifiers│    └───────────┬──────────────┘  │
│                      └──────────────────┘                │                  │
│                                                          ▼                  │
│  ┌────────────────────────────────────────────────────────────────────────┐ │
│  │                      PROOF GENERATION LAYER                            │ │
│  │  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────────────┐│ │
│  │  │ HASH GENERATOR  │  │ ZK PROOF ENGINE │  │ MERKLE TREE BUILDER    ││ │
│  │  │ • SHA-256/384   │  │ • zk-SNARKs     │  │ • Diff verification    ││ │
│  │  │ • Keccak-256    │  │ • Circom/Groth16│  │ • Inclusion proofs     ││ │
│  │  └─────────────────┘  └─────────────────┘  └─────────────────────────┘│ │
│  └────────────────────────────────────────────────────────────────────────┘ │
│                                        │                                    │
│                                        ▼                                    │
│  ┌────────────────────────────────────────────────────────────────────────┐ │
│  │                      BLOCKCHAIN LAYER                                  │ │
│  │  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────────────┐│ │
│  │  │ SMART CONTRACTS │  │ PROOF REGISTRY  │  │ AUDIT TRAIL            ││ │
│  │  │ • Redaction     │  │ • ZK Verifier   │  │ • Immutable logs       ││ │
│  │  │   Notarization  │  │ • Hash Store    │  │ • Timestamp anchors    ││ │
│  │  └─────────────────┘  └─────────────────┘  └─────────────────────────┘│ │
│  └────────────────────────────────────────────────────────────────────────┘ │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Implementation Phases

### Phase 1: Foundation (Weeks 1-2)
- [ ] PHI Detection Engine
- [ ] Basic Redaction System
- [ ] Cryptographic Hash Generation
- [ ] Local Audit Logging

### Phase 2: Blockchain Integration (Weeks 3-4)
- [ ] Smart Contract Development
- [ ] Proof Registry Deployment
- [ ] On-Chain Notarization
- [ ] Hash Anchoring System

### Phase 3: Zero-Knowledge Proofs (Weeks 5-7)
- [ ] ZK Circuit Design
- [ ] Proof Generation Pipeline
- [ ] On-Chain Verification
- [ ] Compliance Proof System

### Phase 4: Production Hardening (Weeks 8-10)
- [ ] Security Audits
- [ ] Performance Optimization
- [ ] Monitoring & Alerting
- [ ] Documentation & Training

---

## Detailed Component Specifications

### 1. PHI Detection Engine

**Location**: `hipaa/detection/`

#### 1.1 Create PHI Pattern Definitions

```python
# hipaa/detection/patterns.py
"""
HIPAA Safe Harbor 18 Identifiers Pattern Matching
"""

PHI_PATTERNS = {
    # Direct Identifiers
    "name": {
        "regex": r"\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)+\b",
        "description": "Full names (First Last)",
        "hipaa_category": "names",
        "risk_level": "high"
    },

    "ssn": {
        "regex": r"\b\d{3}-\d{2}-\d{4}\b",
        "description": "Social Security Numbers",
        "hipaa_category": "social_security",
        "risk_level": "critical"
    },

    "mrn": {
        "regex": r"\b(?:MRN|Medical Record|Patient ID)[:\s#]*(\d{6,12})\b",
        "description": "Medical Record Numbers",
        "hipaa_category": "medical_record",
        "risk_level": "critical"
    },

    "dob": {
        "regex": r"\b(?:DOB|Date of Birth|Born)[:\s]*(\d{1,2}[/-]\d{1,2}[/-]\d{2,4})\b",
        "description": "Dates of Birth",
        "hipaa_category": "dates",
        "risk_level": "high"
    },

    "phone": {
        "regex": r"\b(?:\+1[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b",
        "description": "Phone Numbers",
        "hipaa_category": "telephone",
        "risk_level": "medium"
    },

    "email": {
        "regex": r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b",
        "description": "Email Addresses",
        "hipaa_category": "email",
        "risk_level": "medium"
    },

    "address": {
        "regex": r"\b\d+\s+[\w\s]+(?:Street|St|Avenue|Ave|Road|Rd|Boulevard|Blvd|Drive|Dr|Lane|Ln|Way|Court|Ct)\b",
        "description": "Street Addresses",
        "hipaa_category": "geographic",
        "risk_level": "high"
    },

    "zipcode": {
        "regex": r"\b\d{5}(?:-\d{4})?\b",
        "description": "ZIP Codes (full 5+4 or 5-digit)",
        "hipaa_category": "geographic",
        "risk_level": "medium"
    },

    "ip_address": {
        "regex": r"\b(?:\d{1,3}\.){3}\d{1,3}\b",
        "description": "IP Addresses",
        "hipaa_category": "device_identifiers",
        "risk_level": "medium"
    },

    "health_plan_id": {
        "regex": r"\b(?:Plan|Member|Policy)[:\s#]*([A-Z0-9]{8,20})\b",
        "description": "Health Plan Beneficiary Numbers",
        "hipaa_category": "health_plan",
        "risk_level": "high"
    },

    "vehicle_id": {
        "regex": r"\b[A-HJ-NPR-Z0-9]{17}\b",
        "description": "Vehicle Identification Numbers (VIN)",
        "hipaa_category": "vehicle_identifiers",
        "risk_level": "low"
    },

    "biometric": {
        "regex": r"\b(?:fingerprint|retina|iris|voice|gait|dna)[:\s]+[A-Za-z0-9]+\b",
        "description": "Biometric Identifiers",
        "hipaa_category": "biometric",
        "risk_level": "critical"
    }
}
```

#### 1.2 Create NER-Based Detection

```python
# hipaa/detection/ner_detector.py
"""
Named Entity Recognition for PHI Detection
Uses spaCy + custom medical NER models
"""

from dataclasses import dataclass
from typing import List, Tuple
import spacy

@dataclass
class PHIEntity:
    text: str
    start: int
    end: int
    entity_type: str
    confidence: float
    hipaa_category: str

class NERDetector:
    def __init__(self, model_path: str = "en_core_web_lg"):
        self.nlp = spacy.load(model_path)
        # Add custom medical NER pipeline
        self._add_medical_entities()

    def detect(self, text: str) -> List[PHIEntity]:
        """Detect all PHI entities in text"""
        doc = self.nlp(text)
        entities = []

        for ent in doc.ents:
            if self._is_phi_entity(ent.label_):
                entities.append(PHIEntity(
                    text=ent.text,
                    start=ent.start_char,
                    end=ent.end_char,
                    entity_type=ent.label_,
                    confidence=self._get_confidence(ent),
                    hipaa_category=self._map_to_hipaa(ent.label_)
                ))

        return entities
```

#### 1.3 Create Detection Orchestrator

```python
# hipaa/detection/__init__.py
"""
PHI Detection Orchestrator
Combines regex patterns, NER, and ML models
"""

class PHIDetector:
    def __init__(self, config: DetectionConfig):
        self.pattern_detector = PatternDetector(PHI_PATTERNS)
        self.ner_detector = NERDetector()
        self.ml_detector = MLDetector() if config.use_ml else None
        self.confidence_threshold = config.confidence_threshold

    async def detect_all(self, text: str) -> DetectionResult:
        """
        Run all detection methods and merge results
        """
        results = await asyncio.gather(
            self.pattern_detector.detect(text),
            self.ner_detector.detect(text),
            self.ml_detector.detect(text) if self.ml_detector else []
        )

        merged = self._merge_and_dedupe(results)
        return DetectionResult(
            entities=merged,
            original_text_hash=hashlib.sha256(text.encode()).hexdigest(),
            detection_timestamp=datetime.utcnow()
        )
```

---

### 2. Redaction Engine

**Location**: `hipaa/redaction/`

#### 2.1 Create Redaction Strategies

```python
# hipaa/redaction/strategies.py
"""
Redaction Strategies for Different PHI Types
"""

from enum import Enum
from abc import ABC, abstractmethod

class RedactionMethod(Enum):
    MASK = "mask"              # Replace with [REDACTED]
    HASH = "hash"              # Replace with hash
    TOKENIZE = "tokenize"      # Replace with reversible token
    GENERALIZE = "generalize"  # Age 23 -> Age 20-30
    SUPPRESS = "suppress"      # Remove entirely
    SHIFT = "shift"            # Date shifting

class RedactionStrategy(ABC):
    @abstractmethod
    def redact(self, text: str, entities: List[PHIEntity]) -> RedactionResult:
        pass

class MaskingStrategy(RedactionStrategy):
    """Replace PHI with [REDACTED-TYPE]"""

    def redact(self, text: str, entities: List[PHIEntity]) -> RedactionResult:
        redacted = text
        redaction_map = {}

        # Process in reverse order to maintain positions
        for entity in sorted(entities, key=lambda e: e.start, reverse=True):
            placeholder = f"[REDACTED-{entity.hipaa_category.upper()}]"
            redacted = redacted[:entity.start] + placeholder + redacted[entity.end:]

            redaction_map[entity.start] = {
                "original_hash": hashlib.sha256(entity.text.encode()).hexdigest(),
                "type": entity.hipaa_category,
                "method": "mask"
            }

        return RedactionResult(
            redacted_text=redacted,
            redaction_map=redaction_map,
            original_hash=hashlib.sha256(text.encode()).hexdigest(),
            redacted_hash=hashlib.sha256(redacted.encode()).hexdigest()
        )

class TokenizationStrategy(RedactionStrategy):
    """Replace PHI with reversible tokens (requires secure vault)"""

    def __init__(self, vault: TokenVault):
        self.vault = vault

    def redact(self, text: str, entities: List[PHIEntity]) -> RedactionResult:
        redacted = text
        redaction_map = {}

        for entity in sorted(entities, key=lambda e: e.start, reverse=True):
            token = self.vault.create_token(entity.text, entity.hipaa_category)
            redacted = redacted[:entity.start] + token + redacted[entity.end:]

            redaction_map[entity.start] = {
                "token": token,
                "type": entity.hipaa_category,
                "method": "tokenize"
            }

        return RedactionResult(
            redacted_text=redacted,
            redaction_map=redaction_map,
            reversible=True
        )
```

#### 2.2 Create Redaction Orchestrator

```python
# hipaa/redaction/__init__.py
"""
Redaction Orchestrator
Applies HIPAA-compliant redaction rules
"""

@dataclass
class RedactionConfig:
    """Configuration for redaction behavior"""
    default_method: RedactionMethod = RedactionMethod.MASK
    per_type_methods: Dict[str, RedactionMethod] = field(default_factory=dict)
    generate_proofs: bool = True
    blockchain_enabled: bool = True

class RedactionEngine:
    def __init__(self, config: RedactionConfig):
        self.config = config
        self.strategies = {
            RedactionMethod.MASK: MaskingStrategy(),
            RedactionMethod.HASH: HashingStrategy(),
            RedactionMethod.TOKENIZE: TokenizationStrategy(TokenVault()),
            RedactionMethod.GENERALIZE: GeneralizationStrategy(),
            RedactionMethod.SUPPRESS: SuppressionStrategy(),
        }

    async def redact(
        self,
        text: str,
        entities: List[PHIEntity]
    ) -> RedactionResult:
        """
        Apply redaction according to configuration
        """
        # Group entities by redaction method
        grouped = self._group_by_method(entities)

        # Apply each strategy
        result = RedactionResult(original_text=text)
        for method, method_entities in grouped.items():
            strategy = self.strategies[method]
            partial_result = strategy.redact(result.current_text, method_entities)
            result.merge(partial_result)

        # Generate cryptographic proof
        if self.config.generate_proofs:
            result.proof = await self._generate_proof(text, result)

        return result
```

---

### 3. Cryptographic Proof Generation

**Location**: `hipaa/crypto/`

#### 3.1 Create Hash Generator

```python
# hipaa/crypto/hashing.py
"""
Cryptographic Hashing for Redaction Verification
"""

import hashlib
from typing import List, Tuple
from dataclasses import dataclass

@dataclass
class HashProof:
    original_hash: str           # SHA-256 of original text
    redacted_hash: str           # SHA-256 of redacted text
    merkle_root: str             # Merkle root of all changes
    field_hashes: List[str]      # Individual PHI field hashes
    algorithm: str = "sha256"
    timestamp: str = ""

class HashGenerator:
    """
    Generate cryptographic hashes for redaction verification
    """

    @staticmethod
    def hash_text(text: str, algorithm: str = "sha256") -> str:
        """Generate hash of text content"""
        if algorithm == "sha256":
            return hashlib.sha256(text.encode('utf-8')).hexdigest()
        elif algorithm == "sha384":
            return hashlib.sha384(text.encode('utf-8')).hexdigest()
        elif algorithm == "keccak256":
            # Ethereum-compatible hashing
            from Crypto.Hash import keccak
            k = keccak.new(digest_bits=256)
            k.update(text.encode('utf-8'))
            return k.hexdigest()
        else:
            raise ValueError(f"Unsupported algorithm: {algorithm}")

    @staticmethod
    def generate_merkle_root(hashes: List[str]) -> str:
        """
        Generate Merkle root from list of hashes
        Used for efficient verification of multiple redactions
        """
        if not hashes:
            return hashlib.sha256(b"empty").hexdigest()

        if len(hashes) == 1:
            return hashes[0]

        # Pad to even number
        if len(hashes) % 2 == 1:
            hashes.append(hashes[-1])

        # Build tree bottom-up
        while len(hashes) > 1:
            next_level = []
            for i in range(0, len(hashes), 2):
                combined = hashes[i] + hashes[i+1]
                next_level.append(hashlib.sha256(combined.encode()).hexdigest())
            hashes = next_level

        return hashes[0]

    def generate_proof(
        self,
        original: str,
        redacted: str,
        redacted_fields: List[str]
    ) -> HashProof:
        """
        Generate complete hash proof for redaction
        """
        field_hashes = [self.hash_text(field) for field in redacted_fields]

        return HashProof(
            original_hash=self.hash_text(original),
            redacted_hash=self.hash_text(redacted),
            merkle_root=self.generate_merkle_root(field_hashes),
            field_hashes=field_hashes,
            timestamp=datetime.utcnow().isoformat()
        )
```

#### 3.2 Create Diff Proof Generator

```python
# hipaa/crypto/diff_proof.py
"""
Generate proofs that only PHI was modified
"""

from difflib import SequenceMatcher
from dataclasses import dataclass
from typing import List, Tuple

@dataclass
class DiffProof:
    """Proof that only specific regions were modified"""
    unchanged_regions: List[Tuple[int, int, str]]  # (start, end, hash)
    modified_regions: List[Tuple[int, int, str]]   # (start, end, type)
    modification_merkle_root: str
    unchanged_percentage: float

class DiffProofGenerator:
    """
    Generate cryptographic proofs about what changed and what didn't
    """

    def generate(self, original: str, redacted: str) -> DiffProof:
        """
        Generate proof showing exactly what was modified
        """
        matcher = SequenceMatcher(None, original, redacted)

        unchanged_regions = []
        modified_regions = []

        for op, i1, i2, j1, j2 in matcher.get_opcodes():
            if op == 'equal':
                # Unchanged region - hash it for verification
                unchanged_text = original[i1:i2]
                unchanged_hash = hashlib.sha256(unchanged_text.encode()).hexdigest()
                unchanged_regions.append((i1, i2, unchanged_hash))

            elif op in ('replace', 'delete'):
                # Modified region - record position and type
                modified_regions.append((i1, i2, 'redacted'))

        # Calculate unchanged percentage
        total_len = len(original)
        unchanged_len = sum(end - start for start, end, _ in unchanged_regions)
        unchanged_pct = unchanged_len / total_len if total_len > 0 else 0

        # Generate Merkle root of modifications
        mod_hashes = [
            hashlib.sha256(f"{start}:{end}:{mtype}".encode()).hexdigest()
            for start, end, mtype in modified_regions
        ]
        merkle_root = HashGenerator.generate_merkle_root(mod_hashes)

        return DiffProof(
            unchanged_regions=unchanged_regions,
            modified_regions=modified_regions,
            modification_merkle_root=merkle_root,
            unchanged_percentage=unchanged_pct
        )
```

---

### 4. Zero-Knowledge Proof System

**Location**: `hipaa/zk/`

#### 4.1 ZK Circuit Design

```circom
// hipaa/zk/circuits/redaction_proof.circom
/*
 * Zero-Knowledge Proof: Redaction was performed correctly
 *
 * Public inputs:
 *   - hash_original: Hash of original document
 *   - hash_redacted: Hash of redacted document
 *   - redaction_rules_hash: Hash of approved redaction rules
 *
 * Private inputs:
 *   - original_text: The actual original text (never revealed)
 *   - redacted_text: The actual redacted text
 *   - phi_positions: Positions of PHI in original
 *   - redaction_rules: The rules that were applied
 *
 * Proves:
 *   1. hash_original matches hash(original_text)
 *   2. hash_redacted matches hash(redacted_text)
 *   3. Only positions in phi_positions were modified
 *   4. Modifications followed redaction_rules
 */

pragma circom 2.0.0;

include "poseidon.circom";
include "comparators.circom";

template RedactionProof(maxTextLen, maxRedactions) {
    // Public inputs
    signal input hash_original;
    signal input hash_redacted;
    signal input rules_hash;

    // Private inputs
    signal input original_chars[maxTextLen];
    signal input redacted_chars[maxTextLen];
    signal input phi_starts[maxRedactions];
    signal input phi_ends[maxRedactions];
    signal input phi_types[maxRedactions];

    // Constraint 1: Verify original hash
    component orig_hasher = Poseidon(maxTextLen);
    for (var i = 0; i < maxTextLen; i++) {
        orig_hasher.inputs[i] <== original_chars[i];
    }
    hash_original === orig_hasher.out;

    // Constraint 2: Verify redacted hash
    component redact_hasher = Poseidon(maxTextLen);
    for (var i = 0; i < maxTextLen; i++) {
        redact_hasher.inputs[i] <== redacted_chars[i];
    }
    hash_redacted === redact_hasher.out;

    // Constraint 3: Characters outside PHI regions are unchanged
    component in_phi[maxTextLen];
    for (var i = 0; i < maxTextLen; i++) {
        in_phi[i] = IsInPHIRegion(maxRedactions);
        in_phi[i].position <== i;
        for (var j = 0; j < maxRedactions; j++) {
            in_phi[i].starts[j] <== phi_starts[j];
            in_phi[i].ends[j] <== phi_ends[j];
        }

        // If not in PHI region, chars must match
        (1 - in_phi[i].out) * (original_chars[i] - redacted_chars[i]) === 0;
    }

    // Constraint 4: PHI regions are properly redacted (not just deleted)
    // Verify redaction follows rules (mask pattern, etc.)
    // ... (additional constraints for rule compliance)
}

component main {public [hash_original, hash_redacted, rules_hash]} =
    RedactionProof(10000, 50);
```

#### 4.2 ZK Proof Generator

```python
# hipaa/zk/proof_generator.py
"""
Zero-Knowledge Proof Generation for Redaction Verification
"""

import subprocess
import json
from pathlib import Path
from dataclasses import dataclass
from typing import Optional

@dataclass
class ZKProof:
    proof: dict               # The actual zk-SNARK proof
    public_inputs: List[str]  # Public inputs (hashes only)
    verification_key: str     # Key for on-chain verification
    proof_type: str = "groth16"
    circuit_hash: str = ""    # Hash of the circuit used

class ZKProofGenerator:
    """
    Generate zero-knowledge proofs for HIPAA redaction
    """

    def __init__(self, circuit_path: str, proving_key_path: str):
        self.circuit_path = Path(circuit_path)
        self.proving_key_path = Path(proving_key_path)

    async def generate_proof(
        self,
        original_text: str,
        redacted_text: str,
        phi_entities: List[PHIEntity],
        redaction_rules_hash: str
    ) -> ZKProof:
        """
        Generate a ZK proof that redaction was performed correctly

        The proof demonstrates:
        1. We know the original text (via hash)
        2. We know the redacted text (via hash)
        3. Only PHI positions were modified
        4. Modifications followed approved rules

        Without revealing the actual text content
        """

        # Prepare inputs for the circuit
        inputs = {
            "hash_original": self._text_to_field(original_text),
            "hash_redacted": self._text_to_field(redacted_text),
            "rules_hash": redaction_rules_hash,
            "original_chars": self._text_to_chars(original_text),
            "redacted_chars": self._text_to_chars(redacted_text),
            "phi_starts": [e.start for e in phi_entities],
            "phi_ends": [e.end for e in phi_entities],
            "phi_types": [self._type_to_field(e.hipaa_category) for e in phi_entities]
        }

        # Write inputs to temp file
        input_file = self._write_inputs(inputs)

        # Generate witness
        witness_file = await self._generate_witness(input_file)

        # Generate proof using snarkjs
        proof_result = await self._generate_snark_proof(witness_file)

        return ZKProof(
            proof=proof_result["proof"],
            public_inputs=proof_result["public_inputs"],
            verification_key=self._load_verification_key(),
            circuit_hash=self._get_circuit_hash()
        )

    async def _generate_snark_proof(self, witness_file: Path) -> dict:
        """Generate Groth16 proof using snarkjs"""
        result = subprocess.run([
            "snarkjs", "groth16", "prove",
            str(self.proving_key_path),
            str(witness_file),
            "proof.json",
            "public.json"
        ], capture_output=True, text=True)

        if result.returncode != 0:
            raise RuntimeError(f"Proof generation failed: {result.stderr}")

        with open("proof.json") as f:
            proof = json.load(f)
        with open("public.json") as f:
            public = json.load(f)

        return {"proof": proof, "public_inputs": public}
```

---

### 5. Smart Contract Specifications

**Location**: `hipaa/blockchain/contracts/`

#### 5.1 Redaction Registry Contract

```solidity
// SPDX-License-Identifier: MIT
pragma solidity ^0.8.19;

import "@openzeppelin/contracts/access/AccessControl.sol";
import "@openzeppelin/contracts/utils/cryptography/ECDSA.sol";

/**
 * @title RedactionRegistry
 * @notice Immutable registry of HIPAA redaction proofs
 * @dev Stores only hashes and proofs - NO PHI ever touches the chain
 */
contract RedactionRegistry is AccessControl {
    using ECDSA for bytes32;

    bytes32 public constant NOTARIZER_ROLE = keccak256("NOTARIZER_ROLE");
    bytes32 public constant AUDITOR_ROLE = keccak256("AUDITOR_ROLE");

    struct RedactionRecord {
        bytes32 originalHash;       // SHA-256 of original document
        bytes32 redactedHash;       // SHA-256 of redacted document
        bytes32 proofHash;          // Hash of the ZK proof
        bytes32 rulesHash;          // Hash of redaction rules applied
        uint256 timestamp;          // Block timestamp
        address notarizer;          // Who submitted this record
        bytes32 merkleRoot;         // Merkle root of all redacted fields
        bool verified;              // ZK proof verified on-chain
    }

    // Document ID => Redaction Record
    mapping(bytes32 => RedactionRecord) public records;

    // Notarizer => Document IDs
    mapping(address => bytes32[]) public notarizerRecords;

    // Total records for auditing
    uint256 public totalRecords;

    // Events for audit trail
    event RedactionNotarized(
        bytes32 indexed documentId,
        bytes32 originalHash,
        bytes32 redactedHash,
        address indexed notarizer,
        uint256 timestamp
    );

    event ProofVerified(
        bytes32 indexed documentId,
        bool verified,
        uint256 timestamp
    );

    event AuditRequested(
        bytes32 indexed documentId,
        address indexed auditor,
        uint256 timestamp
    );

    constructor() {
        _grantRole(DEFAULT_ADMIN_ROLE, msg.sender);
        _grantRole(NOTARIZER_ROLE, msg.sender);
    }

    /**
     * @notice Submit a redaction proof to the registry
     * @param documentId Unique identifier for the document
     * @param originalHash Hash of the original document
     * @param redactedHash Hash of the redacted document
     * @param proofHash Hash of the ZK proof
     * @param rulesHash Hash of the redaction rules applied
     * @param merkleRoot Merkle root of all redacted fields
     */
    function notarizeRedaction(
        bytes32 documentId,
        bytes32 originalHash,
        bytes32 redactedHash,
        bytes32 proofHash,
        bytes32 rulesHash,
        bytes32 merkleRoot
    ) external onlyRole(NOTARIZER_ROLE) {
        require(records[documentId].timestamp == 0, "Document already notarized");
        require(originalHash != bytes32(0), "Invalid original hash");
        require(redactedHash != bytes32(0), "Invalid redacted hash");

        records[documentId] = RedactionRecord({
            originalHash: originalHash,
            redactedHash: redactedHash,
            proofHash: proofHash,
            rulesHash: rulesHash,
            timestamp: block.timestamp,
            notarizer: msg.sender,
            merkleRoot: merkleRoot,
            verified: false
        });

        notarizerRecords[msg.sender].push(documentId);
        totalRecords++;

        emit RedactionNotarized(
            documentId,
            originalHash,
            redactedHash,
            msg.sender,
            block.timestamp
        );
    }

    /**
     * @notice Verify a ZK proof on-chain
     * @param documentId The document to verify
     * @param proof The zk-SNARK proof data
     */
    function verifyProof(
        bytes32 documentId,
        bytes calldata proof
    ) external {
        RedactionRecord storage record = records[documentId];
        require(record.timestamp > 0, "Document not found");
        require(!record.verified, "Already verified");

        // Verify the ZK proof using the Groth16 verifier
        bool isValid = _verifyGroth16Proof(
            proof,
            record.originalHash,
            record.redactedHash,
            record.rulesHash
        );

        record.verified = isValid;

        emit ProofVerified(documentId, isValid, block.timestamp);
    }

    /**
     * @notice Request an audit of a document's redaction
     * @param documentId The document to audit
     */
    function requestAudit(bytes32 documentId)
        external
        onlyRole(AUDITOR_ROLE)
    {
        require(records[documentId].timestamp > 0, "Document not found");
        emit AuditRequested(documentId, msg.sender, block.timestamp);
    }

    /**
     * @notice Get complete redaction record
     */
    function getRecord(bytes32 documentId)
        external
        view
        returns (RedactionRecord memory)
    {
        return records[documentId];
    }

    /**
     * @notice Verify a document matches the redacted version on record
     * @param documentId The document ID
     * @param documentHash Hash of the document to verify
     * @return isOriginal True if matches original
     * @return isRedacted True if matches redacted
     */
    function verifyDocument(bytes32 documentId, bytes32 documentHash)
        external
        view
        returns (bool isOriginal, bool isRedacted)
    {
        RedactionRecord storage record = records[documentId];
        return (
            documentHash == record.originalHash,
            documentHash == record.redactedHash
        );
    }

    /**
     * @dev Internal function to verify Groth16 proofs
     */
    function _verifyGroth16Proof(
        bytes calldata proof,
        bytes32 originalHash,
        bytes32 redactedHash,
        bytes32 rulesHash
    ) internal pure returns (bool) {
        // Integration with Groth16 verifier contract
        // This would call the generated verifier from snarkjs
        // Placeholder for actual verification logic
        return true;
    }
}
```

#### 5.2 Audit Trail Contract

```solidity
// SPDX-License-Identifier: MIT
pragma solidity ^0.8.19;

/**
 * @title AuditTrail
 * @notice Immutable audit log for all redaction activities
 */
contract AuditTrail {
    enum EventType {
        DETECTION,      // PHI detected
        REDACTION,      // Redaction performed
        VERIFICATION,   // Proof verified
        ACCESS,         // Document accessed
        EXPORT          // Data exported
    }

    struct AuditEntry {
        bytes32 documentId;
        EventType eventType;
        bytes32 eventHash;      // Hash of event details
        address actor;
        uint256 timestamp;
        uint256 blockNumber;
    }

    // Append-only log
    AuditEntry[] public auditLog;

    // Document ID => Entry indices
    mapping(bytes32 => uint256[]) public documentAudit;

    event AuditEntryAdded(
        uint256 indexed entryId,
        bytes32 indexed documentId,
        EventType eventType,
        address indexed actor
    );

    /**
     * @notice Add an audit entry
     */
    function addEntry(
        bytes32 documentId,
        EventType eventType,
        bytes32 eventHash
    ) external returns (uint256 entryId) {
        entryId = auditLog.length;

        auditLog.push(AuditEntry({
            documentId: documentId,
            eventType: eventType,
            eventHash: eventHash,
            actor: msg.sender,
            timestamp: block.timestamp,
            blockNumber: block.number
        }));

        documentAudit[documentId].push(entryId);

        emit AuditEntryAdded(entryId, documentId, eventType, msg.sender);
    }

    /**
     * @notice Get all audit entries for a document
     */
    function getDocumentAudit(bytes32 documentId)
        external
        view
        returns (AuditEntry[] memory)
    {
        uint256[] storage indices = documentAudit[documentId];
        AuditEntry[] memory entries = new AuditEntry[](indices.length);

        for (uint256 i = 0; i < indices.length; i++) {
            entries[i] = auditLog[indices[i]];
        }

        return entries;
    }

    /**
     * @notice Get total audit entries
     */
    function totalEntries() external view returns (uint256) {
        return auditLog.length;
    }
}
```

---

## Integration Points in Existing Codebase

### Where Blockchain Integration Should Occur

Based on analysis of the current codebase, here are the perfect integration points:

#### 1. Query Pre-Processing (PHI Protection)
**Location**: `community_research_mcp.py:97-144` (search functions)

```python
# BEFORE (current code)
async def search_reddit(query: str, language: str) -> list[dict]:
    # Direct API call with raw query
    ...

# AFTER (with blockchain-verified redaction)
async def search_reddit(query: str, language: str) -> list[dict]:
    # Step 1: Detect PHI in query
    detection_result = await phi_detector.detect_all(query)

    # Step 2: Redact if PHI found
    if detection_result.entities:
        redaction_result = await redaction_engine.redact(
            query,
            detection_result.entities
        )
        query = redaction_result.redacted_text

        # Step 3: Generate and store blockchain proof
        await blockchain_notarizer.notarize(
            original_hash=detection_result.original_text_hash,
            redacted_hash=redaction_result.redacted_hash,
            proof=redaction_result.proof
        )

    # Step 4: Proceed with redacted query
    ...
```

#### 2. Response Post-Processing
**Location**: `api/__init__.py` (aggregate_search)

```python
# Add to aggregate_search() function
async def aggregate_search(...):
    results = await asyncio.gather(...)

    # NEW: Scan results for PHI and redact
    for source, items in results.items():
        for item in items:
            item['snippet'] = await redact_with_proof(item['snippet'])
            item['title'] = await redact_with_proof(item['title'])

    return results
```

#### 3. Audit Logging Enhancement
**Location**: `core/metrics.py`

```python
# Enhance existing metrics with blockchain anchoring
class BlockchainAuditMetrics(APIMetrics):
    async def record_with_blockchain(self, event_type: str, details: dict):
        # Record locally
        super().record_success(...)

        # Anchor to blockchain
        event_hash = hashlib.sha256(
            json.dumps(details, sort_keys=True).encode()
        ).hexdigest()

        await audit_trail_contract.add_entry(
            document_id=details.get('document_id'),
            event_type=event_type,
            event_hash=event_hash
        )
```

#### 4. Caching with Proof Verification
**Location**: `utils/__init__.py`

```python
# Enhance cache to store and verify proofs
def get_cached_result_with_proof(key: str) -> Optional[Tuple[str, bool]]:
    """Get cached result and verify its redaction proof"""
    result = get_cached_result(key)
    if result:
        proof_verified = await verify_cached_proof(key)
        return (result, proof_verified)
    return None
```

#### 5. New Blockchain Module Structure

```
hipaa/
├── __init__.py
├── detection/
│   ├── __init__.py
│   ├── patterns.py          # PHI regex patterns
│   ├── ner_detector.py      # NER-based detection
│   └── ml_detector.py       # ML-based detection
├── redaction/
│   ├── __init__.py
│   ├── strategies.py        # Redaction methods
│   └── rules.py             # HIPAA-compliant rules
├── crypto/
│   ├── __init__.py
│   ├── hashing.py           # Hash generation
│   └── diff_proof.py        # Diff verification
├── zk/
│   ├── __init__.py
│   ├── circuits/            # Circom circuits
│   ├── proof_generator.py   # ZK proof generation
│   └── verifier.py          # Proof verification
└── blockchain/
    ├── __init__.py
    ├── contracts/           # Solidity contracts
    ├── notarizer.py         # On-chain notarization
    ├── audit_trail.py       # Audit logging
    └── web3_client.py       # Web3 connection
```

---

## Testing Strategy

### Unit Tests

```python
# tests/test_phi_detection.py
class TestPHIDetection:
    def test_ssn_detection(self):
        text = "Patient SSN: 123-45-6789"
        result = detector.detect(text)
        assert len(result.entities) == 1
        assert result.entities[0].hipaa_category == "social_security"

    def test_no_false_positives(self):
        text = "The year 2023-2024 was productive"
        result = detector.detect(text)
        assert len(result.entities) == 0

# tests/test_redaction.py
class TestRedaction:
    def test_masking_preserves_structure(self):
        text = "John Doe, DOB: 01/15/1980"
        result = redactor.redact(text)
        assert "[REDACTED-NAME]" in result.redacted_text
        assert "[REDACTED-DATE]" in result.redacted_text
        assert "1980" not in result.redacted_text

# tests/test_blockchain.py
class TestBlockchainProof:
    def test_proof_generation(self):
        proof = generator.generate_proof(original, redacted, entities)
        assert proof.original_hash != proof.redacted_hash
        assert len(proof.merkle_root) == 64

    def test_proof_verification(self):
        proof = generator.generate_proof(...)
        is_valid = verifier.verify(proof)
        assert is_valid == True
```

### Integration Tests

```python
# tests/test_integration.py
class TestEndToEndRedaction:
    async def test_full_pipeline(self):
        # Input with PHI
        query = "Patient John Doe SSN 123-45-6789 needs help"

        # Run through system
        result = await community_search(query=query, language="python")

        # Verify PHI removed from all results
        for source, items in result['results'].items():
            for item in items:
                assert "John Doe" not in item['snippet']
                assert "123-45-6789" not in item['snippet']

        # Verify blockchain proof exists
        assert result['audit']['blockchain_proof'] is not None
        assert result['audit']['blockchain_tx'] is not None
```

---

## Deployment & Operations

### Infrastructure Requirements

```yaml
# docker-compose.yml
version: '3.8'
services:
  redaction-service:
    build: .
    environment:
      - BLOCKCHAIN_RPC_URL=https://polygon-mainnet.infura.io/v3/KEY
      - REDIS_URL=redis://redis:6379
      - ZK_PROVING_KEY_PATH=/keys/proving.key
    volumes:
      - ./keys:/keys:ro

  blockchain-listener:
    build: ./blockchain
    environment:
      - CONTRACT_ADDRESS=0x...

  redis:
    image: redis:7-alpine
    volumes:
      - redis-data:/data

  postgres:
    image: postgres:15
    environment:
      - POSTGRES_DB=hipaa_audit
```

### Monitoring

```python
# hipaa/monitoring/alerts.py
class RedactionAlerts:
    async def check_proof_failures(self):
        """Alert on proof generation failures"""
        failures = await metrics.get_proof_failures(last_hour=True)
        if failures > threshold:
            await notify_oncall("High proof failure rate")

    async def check_blockchain_sync(self):
        """Alert on blockchain sync issues"""
        last_block = await blockchain.get_last_synced_block()
        current_block = await blockchain.get_current_block()
        if current_block - last_block > 100:
            await notify_oncall("Blockchain sync delayed")
```

---

## Compliance Verification

### HIPAA Safe Harbor Verification Checklist

| PHI Category | Detection | Redaction | Proof | Status |
|--------------|-----------|-----------|-------|--------|
| Names | ✅ | ✅ | ✅ | Required |
| Geographic (smaller than state) | ✅ | ✅ | ✅ | Required |
| Dates (except year) | ✅ | ✅ | ✅ | Required |
| Phone numbers | ✅ | ✅ | ✅ | Required |
| Fax numbers | ✅ | ✅ | ✅ | Required |
| Email addresses | ✅ | ✅ | ✅ | Required |
| SSN | ✅ | ✅ | ✅ | Required |
| Medical record numbers | ✅ | ✅ | ✅ | Required |
| Health plan beneficiary numbers | ✅ | ✅ | ✅ | Required |
| Account numbers | ✅ | ✅ | ✅ | Required |
| Certificate/license numbers | ✅ | ✅ | ✅ | Required |
| Vehicle identifiers | ✅ | ✅ | ✅ | Required |
| Device identifiers | ✅ | ✅ | ✅ | Required |
| Web URLs | ✅ | ✅ | ✅ | Required |
| IP addresses | ✅ | ✅ | ✅ | Required |
| Biometric identifiers | ✅ | ✅ | ✅ | Required |
| Full-face photographs | ⚠️ | ⚠️ | ✅ | Image processing needed |
| Unique identifying codes | ✅ | ✅ | ✅ | Required |

---

## Implementation Checklist

### Phase 1: Foundation

- [ ] **1.1 PHI Detection Engine**
  - [ ] Create `hipaa/detection/patterns.py` with all 18 HIPAA identifiers
  - [ ] Create `hipaa/detection/ner_detector.py` with spaCy integration
  - [ ] Create `hipaa/detection/ml_detector.py` for edge cases
  - [ ] Create `hipaa/detection/__init__.py` orchestrator
  - [ ] Write unit tests for all detection patterns
  - [ ] Test against HIPAA test dataset

- [ ] **1.2 Redaction Engine**
  - [ ] Create `hipaa/redaction/strategies.py` with all methods
  - [ ] Create `hipaa/redaction/rules.py` with HIPAA-compliant rules
  - [ ] Create `hipaa/redaction/__init__.py` orchestrator
  - [ ] Implement reversible tokenization with secure vault
  - [ ] Write unit tests for all redaction strategies
  - [ ] Test redaction preserves document structure

- [ ] **1.3 Cryptographic Hashing**
  - [ ] Create `hipaa/crypto/hashing.py` with SHA-256/Keccak
  - [ ] Create `hipaa/crypto/diff_proof.py` for change verification
  - [ ] Implement Merkle tree generation
  - [ ] Write unit tests for hash consistency
  - [ ] Test Merkle proof verification

- [ ] **1.4 Local Audit Logging**
  - [ ] Enhance `core/metrics.py` with redaction events
  - [ ] Create audit log rotation and retention
  - [ ] Implement log integrity verification
  - [ ] Write tests for audit completeness

### Phase 2: Blockchain Integration

- [ ] **2.1 Smart Contract Development**
  - [ ] Develop `RedactionRegistry.sol` contract
  - [ ] Develop `AuditTrail.sol` contract
  - [ ] Write comprehensive Solidity tests
  - [ ] Audit contracts for security vulnerabilities
  - [ ] Deploy to testnet (Polygon Mumbai)

- [ ] **2.2 Web3 Client**
  - [ ] Create `hipaa/blockchain/web3_client.py`
  - [ ] Implement transaction retry logic
  - [ ] Implement gas estimation and management
  - [ ] Create event listener for audit events
  - [ ] Write integration tests with testnet

- [ ] **2.3 Notarization Service**
  - [ ] Create `hipaa/blockchain/notarizer.py`
  - [ ] Implement batch notarization for efficiency
  - [ ] Implement proof caching to reduce gas costs
  - [ ] Create fallback for blockchain unavailability
  - [ ] Write end-to-end notarization tests

### Phase 3: Zero-Knowledge Proofs

- [ ] **3.1 ZK Circuit Development**
  - [ ] Design redaction proof circuit in Circom
  - [ ] Implement constraints for PHI region verification
  - [ ] Implement constraints for rule compliance
  - [ ] Generate trusted setup (Powers of Tau)
  - [ ] Generate proving and verification keys

- [ ] **3.2 ZK Proof Generator**
  - [ ] Create `hipaa/zk/proof_generator.py`
  - [ ] Integrate with snarkjs for proof generation
  - [ ] Implement witness generation
  - [ ] Optimize proof generation time
  - [ ] Write tests for proof validity

- [ ] **3.3 On-Chain Verification**
  - [ ] Generate Solidity verifier from circuit
  - [ ] Deploy verifier contract
  - [ ] Integrate verification into RedactionRegistry
  - [ ] Test end-to-end verification flow
  - [ ] Benchmark gas costs for verification

### Phase 4: Codebase Integration

- [ ] **4.1 Query Pre-Processing**
  - [ ] Add PHI detection before all search functions
  - [ ] Integrate redaction into `community_research_mcp.py`
  - [ ] Add blockchain proof generation for queries
  - [ ] Update all API modules in `api/` directory
  - [ ] Test query redaction doesn't break search results

- [ ] **4.2 Response Post-Processing**
  - [ ] Add PHI scanning to `aggregate_search()`
  - [ ] Implement result redaction pipeline
  - [ ] Generate proofs for redacted responses
  - [ ] Update response format to include proof metadata
  - [ ] Test response redaction accuracy

- [ ] **4.3 Cache Enhancement**
  - [ ] Store proofs with cached results in `utils/`
  - [ ] Implement proof verification on cache retrieval
  - [ ] Add proof expiration handling
  - [ ] Test cache integrity with proofs

- [ ] **4.4 Metrics Enhancement**
  - [ ] Add blockchain anchoring to `core/metrics.py`
  - [ ] Create redaction-specific metrics
  - [ ] Implement proof generation metrics
  - [ ] Add blockchain transaction tracking

### Phase 5: Production Hardening

- [ ] **5.1 Security Audit**
  - [ ] Conduct code security review
  - [ ] Audit smart contracts (external audit)
  - [ ] Penetration testing
  - [ ] HIPAA compliance review
  - [ ] Document all security measures

- [ ] **5.2 Performance Optimization**
  - [ ] Profile PHI detection performance
  - [ ] Optimize ZK proof generation (target < 5s)
  - [ ] Implement batch blockchain transactions
  - [ ] Add caching for repeated patterns
  - [ ] Benchmark end-to-end latency

- [ ] **5.3 Monitoring & Alerting**
  - [ ] Create Grafana dashboards for redaction metrics
  - [ ] Set up alerts for proof failures
  - [ ] Monitor blockchain sync status
  - [ ] Create audit report generation
  - [ ] Implement anomaly detection

- [ ] **5.4 Documentation**
  - [ ] Write API documentation
  - [ ] Create deployment guide
  - [ ] Write operator runbook
  - [ ] Document compliance procedures
  - [ ] Create training materials

### Phase 6: Deployment

- [ ] **6.1 Infrastructure Setup**
  - [ ] Deploy to staging environment
  - [ ] Configure blockchain connections
  - [ ] Set up secure key management
  - [ ] Configure backup and recovery
  - [ ] Test disaster recovery

- [ ] **6.2 Mainnet Deployment**
  - [ ] Deploy contracts to mainnet (Polygon)
  - [ ] Verify contracts on block explorer
  - [ ] Configure production RPC endpoints
  - [ ] Set up gas management
  - [ ] Monitor first 24 hours

- [ ] **6.3 Go-Live Verification**
  - [ ] Run full integration test suite
  - [ ] Verify all proofs are generated
  - [ ] Confirm blockchain anchoring works
  - [ ] Validate audit trail completeness
  - [ ] Sign-off from compliance team

---

## Summary: The Value of Blockchain for HIPAA Redaction

### What You Gain

1. **Cryptographic Proof of Correct Redaction**
   - Prove exactly what was removed without revealing it
   - Mathematical certainty, not just policy

2. **Immutable Audit Trail**
   - Cannot be altered or deleted
   - Provides legal defensibility

3. **Trustless Verification**
   - Any party can verify redaction correctness
   - No need for trusted intermediaries

4. **Zero-Knowledge Privacy**
   - Prove compliance without exposing PHI
   - Perfect for cross-institution sharing

5. **Regulatory Compliance**
   - Satisfies HIPAA audit requirements
   - Provides evidence for OCR investigations

### What It Doesn't Replace

- PHI detection still requires ML/NLP
- Redaction logic still runs off-chain
- Clinical content validation is separate
- Human oversight is still necessary

---

## Appendix: File Structure After Implementation

```
community-research-mcp/
├── community_research_mcp.py    # Enhanced with PHI protection
├── api/                         # Search APIs with redaction hooks
│   ├── __init__.py             # aggregate_search with PHI scanning
│   └── ...
├── core/                        # Enhanced with blockchain metrics
│   ├── metrics.py              # + blockchain anchoring
│   └── ...
├── hipaa/                       # NEW: HIPAA compliance module
│   ├── __init__.py
│   ├── detection/
│   │   ├── __init__.py
│   │   ├── patterns.py
│   │   ├── ner_detector.py
│   │   └── ml_detector.py
│   ├── redaction/
│   │   ├── __init__.py
│   │   ├── strategies.py
│   │   └── rules.py
│   ├── crypto/
│   │   ├── __init__.py
│   │   ├── hashing.py
│   │   └── diff_proof.py
│   ├── zk/
│   │   ├── __init__.py
│   │   ├── circuits/
│   │   │   └── redaction_proof.circom
│   │   ├── proof_generator.py
│   │   └── verifier.py
│   └── blockchain/
│       ├── __init__.py
│       ├── contracts/
│       │   ├── RedactionRegistry.sol
│       │   └── AuditTrail.sol
│       ├── notarizer.py
│       ├── audit_trail.py
│       └── web3_client.py
├── tests/
│   ├── test_phi_detection.py
│   ├── test_redaction.py
│   ├── test_crypto.py
│   ├── test_zk_proofs.py
│   └── test_blockchain.py
└── docs/
    ├── HIPAA_COMPLIANCE.md
    ├── BLOCKCHAIN_ARCHITECTURE.md
    └── DEPLOYMENT_GUIDE.md
```

---

*This document serves as the comprehensive implementation guide for integrating blockchain-powered HIPAA redaction into the Community Research MCP server. The blockchain acts as a tamper-proof notary, proving redaction correctness without ever exposing PHI.*
