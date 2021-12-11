import sys
import os
import urllib.request
import tarfile

IS_WIN = os.name == 'nt'


def non_win_string(s):
    return s if not IS_WIN else ''


CONSOLE_CLEAR = non_win_string('\033[0;0m')
CONSOLE_BOLD = non_win_string('\033[0;1m')
CONSOLE_WAIT = non_win_string('⏳')
CONSOLE_DONE = non_win_string('✅')
CONSOLE_STARS = non_win_string('✨')
CONSOLE_ERROR = non_win_string('❌')


def print_usage():
    print('Usage:')
    print('')
    print(
        f'{CONSOLE_BOLD}python -m laserembeddings download-models [OUTPUT_DIRECTORY]{CONSOLE_CLEAR}'
    )
    print(
        '   Downloads LASER model files. If OUTPUT_DIRECTORY is omitted,'
        '\n'
        f'   the models will be placed into the {CONSOLE_BOLD}data{CONSOLE_CLEAR} directory of the module'
    )
    print('')
    print(
        f'{CONSOLE_BOLD}python -m laserembeddings download-test-data{CONSOLE_CLEAR}'
    )
    print('   downloads data needed to run the tests')
    print('')


def download_file(url, dest):
    print(f'{CONSOLE_WAIT}   Downloading {url}...', end='')
    sys.stdout.flush()
    urllib.request.urlretrieve(url, dest)
    print(f'\r{CONSOLE_DONE}   Downloaded {url}    ')


def extract_tar(tar, output_dir):
    print(f'{CONSOLE_WAIT}   Extracting archive...', end='')
    sys.stdout.flush()
    with tarfile.open(tar) as t:
        t.extractall(output_dir)
    print(f'\r{CONSOLE_DONE}   Extracted archive    ')


def download_models(output_dir):
    print(f'Downloading models into {output_dir}')
    print('')

    download_file('https://dl.fbaipublicfiles.com/laser/models/93langs.fcodes',
                  os.path.join(output_dir, '93langs.fcodes'))
    download_file('https://dl.fbaipublicfiles.com/laser/models/93langs.fvocab',
                  os.path.join(output_dir, '93langs.fvocab'))
    download_file(
        'https://dl.fbaipublicfiles.com/laser/models/bilstm.93langs.2018-12-26.pt',
        os.path.join(output_dir, 'bilstm.93langs.2018-12-26.pt'))

    print('')
    print(f'{CONSOLE_STARS} You\'re all set!')


def download_and_extract_test_data(output_dir):
    print(f'Downloading test data into {output_dir}')
    print('')

    download_file(
        'https://github.com/yannvgn/laserembeddings-test-data/releases/download/v1.0.2/laserembeddings-test-data.tar.gz',
        os.path.join(output_dir, 'laserembeddings-test-data.tar.gz'))

    extract_tar(os.path.join(output_dir, 'laserembeddings-test-data.tar.gz'),
                output_dir)

    print('')
    print(f'{CONSOLE_STARS} Ready to test all that!')


def main():
    if len(sys.argv) == 1:
        print_usage()
        return

    if any(arg == '--help' for arg in sys.argv):
        print_usage()
        return

    if sys.argv[1] == 'download-models':
        output_dir = sys.argv[2] if len(sys.argv) > 2 else os.path.join(
            os.path.dirname(os.path.realpath(__file__)), 'data')

        download_models(output_dir)

    elif sys.argv[1] == 'download-test-data':
        if len(sys.argv) > 2:
            print_usage()
            return

        repository_root = os.path.dirname(
            os.path.dirname(os.path.realpath(__file__)))

        if not os.path.isfile(os.path.join(repository_root, 'pyproject.toml')):
            print(
                f"{CONSOLE_ERROR}  Looks like you're not running laserembeddings from its source code"
            )
            print(
                "     → please checkout https://github.com/yannvgn/laserembeddings.git"
            )
            print(
                '       then run "python -m laserembeddings download-test-data" from the root of the repository'
            )
            return

        download_and_extract_test_data(
            os.path.join(repository_root, 'tests', 'data'))


if __name__ == "__main__":
    main()
