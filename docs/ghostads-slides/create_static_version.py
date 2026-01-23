#!/usr/bin/env python3
import re
import os

def remove_pauses_and_overlays(content):
    """Remove all pause commands and overlay specifications from LaTeX content."""
    # Remove \pause commands
    content = re.sub(r'\\pause\s*', '', content)

    # Remove overlay specifications like <2->, <3-5>, etc.
    content = re.sub(r'<[0-9,\-]+>', '', content)

    # Replace \visible<...>{...} with just the content
    content = re.sub(r'\\visible<[^>]*>\s*{([^{}]*(?:{[^{}]*}[^{}]*)*)}', r'\1', content)

    # Replace \only<...>{...} with just the content (keep everything)
    content = re.sub(r'\\only<[^>]*>\s*{([^{}]*(?:{[^{}]*}[^{}]*)*)}', r'\1', content)

    # Replace \uncover<...>{...} with just the content
    content = re.sub(r'\\uncover<[^>]*>\s*{([^{}]*(?:{[^{}]*}[^{}]*)*)}', r'\1', content)

    # Remove \onslide<...> commands
    content = re.sub(r'\\onslide<[^>]*>\s*', '', content)

    # Remove \alert<...>{...} overlay specifications but keep \alert{...}
    content = re.sub(r'\\alert<[^>]*>', r'\\alert', content)

    # Remove any remaining overlay specifications on items
    content = re.sub(r'\\item<[^>]*>', r'\\item', content)

    return content

def process_file(input_path, output_path):
    """Process a single file to remove pauses and overlays."""
    with open(input_path, 'r', encoding='utf-8') as f:
        content = f.read()

    modified_content = remove_pauses_and_overlays(content)

    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(modified_content)

    return output_path

def main():
    slides_dir = '/Users/pranjal/Code/marketplace-incrementality/ghostads/slides'

    # Files to process
    files_to_process = [
        'main.tex',
        'part1_problem.tex',
        'part2_traditional.tex',
        'part3_ghostads.tex',
        'part4_implementation.tex',
        'part5_extensions.tex',
        'backup_slides.tex'
    ]

    # Create static versions
    for filename in files_to_process:
        input_path = os.path.join(slides_dir, filename)

        # Create output filename
        if filename == 'main.tex':
            output_filename = 'main_static.tex'
        else:
            base, ext = os.path.splitext(filename)
            output_filename = f'{base}_static{ext}'

        output_path = os.path.join(slides_dir, output_filename)

        if os.path.exists(input_path):
            process_file(input_path, output_path)
            print(f"Created: {output_filename}")
        else:
            print(f"Warning: {filename} not found")

    # Update main_static.tex to use the static versions of input files
    main_static_path = os.path.join(slides_dir, 'main_static.tex')
    with open(main_static_path, 'r', encoding='utf-8') as f:
        main_content = f.read()

    # Replace input file references
    main_content = main_content.replace('\\input{part1_problem.tex}', '\\input{part1_problem_static.tex}')
    main_content = main_content.replace('\\input{part2_traditional.tex}', '\\input{part2_traditional_static.tex}')
    main_content = main_content.replace('\\input{part3_ghostads.tex}', '\\input{part3_ghostads_static.tex}')
    main_content = main_content.replace('\\input{part4_implementation.tex}', '\\input{part4_implementation_static.tex}')
    main_content = main_content.replace('\\input{part5_extensions.tex}', '\\input{part5_extensions_static.tex}')
    main_content = main_content.replace('\\input{backup_slides.tex}', '\\input{backup_slides_static.tex}')

    with open(main_static_path, 'w', encoding='utf-8') as f:
        f.write(main_content)

    print("\nStatic version created successfully!")
    print("You can now compile main_static.tex")

if __name__ == '__main__':
    main()