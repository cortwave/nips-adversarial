from os import listdir
from subprocess import call


def main(start_dir='sample_defenses/'):
    commands = []
    dirs = filter(lambda x: not x.endswith('.sh'), listdir(start_dir))
    for d in dirs:
        d = './' + start_dir + d + '/'
        print(d)

        call(f'rm *.zip', shell=True, cwd=d)
        call(f'zip result.zip * ', shell=True, cwd=d)
        command = f'python validation_tool/validate_submission.py --submission_filename={d}result.zip --submission_type=defense --use_gpu'
        commands.append(command)

    call(' && '.join(commands))

if __name__ == '__main__':
    main()
