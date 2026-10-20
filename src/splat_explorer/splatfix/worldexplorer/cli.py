"""Independent starter CLI, with jobs recoverable after laptop restart."""
import argparse
import json
from pathlib import Path
from . import DEFAULT_PROMPT
from .service import WorldExplorerService


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, default=Path('outputs/worldexplorer'))
    sub = parser.add_subparsers(dest='command', required=True)
    sub.add_parser('setup')
    sub.add_parser('list')
    generate = sub.add_parser('generate')
    generate.add_argument('prompt', nargs='?', default=DEFAULT_PROMPT)
    generate.add_argument('--mode', choices=('automatic','fast','manual'), default='automatic')
    generate.add_argument('--translation-scaling',type=float,default=3.0)
    generate.add_argument('--num-images-for-vggt',type=int,default=40)
    for name in ('status','download','resume'):
        sub.add_parser(name).add_argument('id')
    expand = sub.add_parser('expand')
    expand.add_argument('id')
    expand.add_argument('--selection', type=Path, required=True, help='JSON object mapping 001/003/005/007 to candidate paths')
    args = parser.parse_args()
    service = WorldExplorerService(args.output)
    if args.command == 'list': result = service.jobs()
    elif args.command == 'status': result = service.refresh(args.id)
    elif args.command == 'expand': result = service.expand(args.id, json.loads(args.selection.read_text()))
    elif args.command == 'download': result = service.download(args.id)
    elif args.command == 'resume': result = service.resume(args.id)
    else:
        result = service.create({**vars(args), 'operation':args.command})
    print(json.dumps(result,indent=2))

if __name__ == '__main__':
    main()
