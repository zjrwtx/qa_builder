from Bio.SeqUtils import molecular_weight
from Bio.Seq import Seq
# Removed Bio.Alphabet import as it's deprecated

def get_amino_acid_mass():
    """Returns a dictionary of amino acid masses."""
    masses = {
        'G': 57, 'A': 71, 'S': 87, 'P': 97, 'V': 99, 'T': 101, 'C': 103,
        'I': 113, 'L': 113, 'N': 114, 'D': 115, 'K': 128, 'Q': 128, 'E': 129,
        'M': 131, 'H': 137, 'F': 147, 'R': 156, 'Y': 163, 'W': 186
    }
    return masses

def get_linear_spectrum(peptide):
    """
    Generate the linear spectrum of a peptide.
    
    Args:
        peptide: A string of amino acids.
        
    Returns:
        A list of integers representing the linear spectrum.
    """
    amino_acid_mass = get_amino_acid_mass()
    prefix_mass = [0]
    
    # Calculate prefix masses
    for i in range(len(peptide)):
        prefix_mass.append(prefix_mass[i] + amino_acid_mass[peptide[i]])
    
    # Generate spectrum by considering all subpeptides
    linear_spectrum = [0]  # Add mass of empty peptide
    
    for i in range(len(peptide)):
        for j in range(i + 1, len(peptide) + 1):
            linear_spectrum.append(prefix_mass[j] - prefix_mass[i])
    
    return sorted(linear_spectrum)

def linear_score(peptide, spectrum):
    """
    Compute the linear score of a peptide against a spectrum.
    
    Args:
        peptide: A string of amino acids.
        spectrum: A list of integers representing a spectrum.
        
    Returns:
        The score indicating how well the theoretical spectrum of the peptide
        matches the experimental spectrum.
    """
    theoretical_spectrum = get_linear_spectrum(peptide)
    
    # Convert to dictionaries with counts
    theoretical_counts = {}
    for mass in theoretical_spectrum:
        theoretical_counts[mass] = theoretical_counts.get(mass, 0) + 1
        
    experimental_counts = {}
    for mass in spectrum:
        experimental_counts[mass] = experimental_counts.get(mass, 0) + 1
    
    # Calculate score
    score = 0
    for mass, count in theoretical_counts.items():
        common_count = min(count, experimental_counts.get(mass, 0))
        score += common_count
        
    return score

def main():
    # Parse input from qa_data.json
    peptide = "NQEL"
    spectrum = [0, 99, 113, 114, 128, 227, 257, 299, 355, 356, 370, 371, 484]
    
    # Compute score
    score = linear_score(peptide, spectrum)
    print(f"LinearScore({peptide}, {spectrum}) = {score}")
    
    return score

if __name__ == "__main__":
    result = main()
    print(f"Final answer: {result}")