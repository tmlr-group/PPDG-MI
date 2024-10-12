def convert_names(full_name):
    # Split the names by ' and ' to handle multiple authors
    authors = full_name.split(' and ')
    initials = []

    for author in authors:
        # Check if the author is "others"
        if author.strip().lower() == "others":
            initials.append("others")
        else:
            # Split each author's name into last name and first name
            last_name, first_name = author.split(', ')
            # Append the last name and the first initial
            initials.append(f"{last_name}, {first_name[0]}.")

    # Join the processed names back together
    return ' and '.join(initials)

# Example usage
full_name = '''Wu, Tailin and Ren, Hongyu and Li, Pan and Leskovec, Jure
'''

converted_name = convert_names(full_name)
print(converted_name)  # Output: Radford, A. and Wu, J.